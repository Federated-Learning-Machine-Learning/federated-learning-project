import flwr as fl
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from typing import Callable, Dict, List
from flwr.client import ClientApp
from flwr.common import Context
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from enum import Enum
from torch import device as torch_device
from flwr.client import Client
from torch.cuda.amp import GradScaler, autocast
from editing import SparseSGDM, TaLoSPruner, iterative_pruning
from torch.optim import Optimizer

class OptimizerType(Enum):
    SSGD = "ssgd"
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

class SchedulerType(Enum):
    COSINE = "cosine"
    COSINE_RESTART = "cosine_restart"
    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    CONSTANT = "constant"
    LINEAR = "linear"

def build_client_fn(
    use_iid: bool,
    optimizer_type: OptimizerType,
    scheduler_type: SchedulerType,
    optimizer_config: Dict,
    scheduler_config: Dict,
    iid_partitions: List[Dataset],
    non_iid_partitions: List[Dataset],
    model_fn: Callable[[], nn.Module],
    device: torch_device,
    valset: Dataset,
    local_epochs: int = 1,
) -> Callable[[Context], Client]:
    """
    Build a Flower-compatible client function with configurable optimizer and scheduler.

    Args:
        use_iid (bool): Whether to use IID (True) or non-IID (False) training data.
        optimizer_type (OptimizerType): Enum specifying the optimizer to use.
        scheduler_type (SchedulerType): Enum specifying the scheduler to use.
        optimizer_config (Dict): Dictionary of optimizer hyperparameters (e.g., lr, momentum).
        scheduler_config (Dict): Dictionary of scheduler hyperparameters (e.g., T_max, gamma).
        iid_partitions (List[Dataset]): List of IID-partitioned datasets (one per client).
        non_iid_partitions (List[Dataset]): List of non-IID-partitioned datasets.
        model_fn (Callable): Function that returns a fresh model instance.
        device (torch.device): PyTorch device (CPU or CUDA).
        valset (Dataset): Shared global validation dataset.
        local_epochs (int): Number of local epochs to perform.

    Returns:
        Callable[[Context], Client]: A function that returns a Flower client instance when called with a Context.
    """

    def make_optimizer(model: nn.Module, optimizer_type: OptimizerType, optimizer_config: dict) -> optim.Optimizer:
        if optimizer_type == OptimizerType.SGD:
            return optim.SGD(model.parameters(), **optimizer_config)
        elif optimizer_type == OptimizerType.ADAM:
            return optim.Adam(model.parameters(), **optimizer_config)
        elif optimizer_type == OptimizerType.ADAMW:
            return optim.AdamW(model.parameters(), **optimizer_config)
        elif optimizer_type == OptimizerType.RMSPROP:
            return optim.RMSprop(model.parameters(), **optimizer_config)
        elif optimizer_type == OptimizerType.ADAGRAD:
            return optim.Adagrad(model.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: SchedulerType, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
        if scheduler_type == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.COSINE_RESTART:
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.MULTISTEP:
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.EXPONENTIAL:
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.CONSTANT:
            return optim.lr_scheduler.ConstantLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.LINEAR:
            return optim.lr_scheduler.LinearLR(optimizer, **scheduler_config)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    def client_fn(context: Context) -> Client:
        """
        Construct a client instance based on the context provided by Flower.

        Args:
            context (Context): Context object including the client's partition ID.

        Returns:
            fl.client.Client: A fully initialized Flower client.
        """
        cid = int(context.node_config["partition-id"])
        print(f"LOG: Initializing client with CID={cid}")
        
        trainset = iid_partitions[cid] if use_iid else non_iid_partitions[cid]
        print(f"{cid}-LOG: Data partition assigned to client {cid} -> {'IID' if use_iid else 'Non-IID'}")

        model = model_fn()
        print(f"{cid}-LOG: Model initialized for client {cid}")
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        valloader = DataLoader(valset, batch_size=64)
        print(f"{cid}-LOG: Dataloaders initialized for client {cid}")

        return CIFARFederatedClient(
            cid=cid,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            optimizer_config=optimizer_config,
            optimizer_type=optimizer_type,
            scheduler_config=scheduler_config,
            scheduler_type=scheduler_type,
            optimizer_fn=make_optimizer,
            scheduler_fn=make_scheduler,
            local_epochs=local_epochs
        ).to_client()
        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")

    return client_fn

def build_client_fn_model_editing(
    use_iid: bool,
    optimizer_type: OptimizerType,
    scheduler_type: SchedulerType,
    optimizer_config: Dict,
    scheduler_config: Dict,
    iid_partitions: List[Dataset],
    non_iid_partitions: List[Dataset],
    model_fn: Callable[[], nn.Module],
    device: torch.device,
    valset: Dataset,
    local_epochs: int = 1,
    pruning_rounds: int = 4,
    final_sparsity: float = 0.9,
    num_batches: int = 3,
    momentum: float = 0.9,
    weight_decay: float = 0.0005
) -> Callable[[Context], Client]:

    def make_optimizer(
        model: torch.nn.Module,
        optimizer_type: str,
        optimizer_config: dict,
        masks: list = None
    ) -> Optimizer:
        if optimizer_type == OptimizerType.SSGD:
            return SparseSGDM(
                model.parameters(),
                lr=optimizer_config.get("lr", 0.01),
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=optimizer_config.get("weight_decay", 0.0005),
                masks=masks
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
        if scheduler_type == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.COSINE_RESTART:
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.MULTISTEP:
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.EXPONENTIAL:
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.CONSTANT:
            return optim.lr_scheduler.ConstantLR(optimizer, **scheduler_config)
        elif scheduler_type == SchedulerType.LINEAR:
            return optim.lr_scheduler.LinearLR(optimizer, **scheduler_config)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        print(f"LOG: Initializing client with CID={cid}")

        trainset = iid_partitions[cid] if use_iid else non_iid_partitions[cid]
        print(f"{cid}-LOG: Data partition assigned to client {cid} -> {'IID' if use_iid else 'Non-IID'}")


        model = model_fn()
        print(f"{cid}-LOG: Model initialized for client {cid}")
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        valloader = DataLoader(valset, batch_size=64)
        print(f"{cid}-LOG: Dataloaders initialized for client {cid}")

        try:
            return CIFARFederatedClientModelEditing(
                cid=cid,
                model=model,
                trainloader=trainloader,
                valloader=valloader,
                device=device,
                optimizer_fn=make_optimizer,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                scheduler_fn=make_scheduler,
                local_epochs=local_epochs,
                pruning_rounds=pruning_rounds,
                final_sparsity=final_sparsity,
                num_batches=num_batches,
                momentum=momentum,
                weight_decay=weight_decay
            ).to_client()
            print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")
        except Exception as e:
            raise e

    return client_fn


class CIFARFederatedClient(fl.client.NumPyClient):
    """
    A flexible Flower federated client for CIFAR-based image classification.

    This client requires explicit definitions for both optimizer and scheduler,
    allowing full control of local training behavior in FL experiments.
    """

    def __init__(
        self,
        cid,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer_type: OptimizerType,
        scheduler_type: SchedulerType,
        optimizer_config: Dict,
        scheduler_config: Dict,
        optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        local_epochs: int,
    ):
        """
        Initialize the federated client.

        Args:
            cid: Client id.
            model (nn.Module): PyTorch model instance.
            trainloader (DataLoader): Training dataloader.
            valloader (DataLoader): Validation dataloader.
            device (torch.device): Device to run training on (CPU or CUDA).
            optimizer_type (OptimizerType): Enum specifying the optimizer to use.
            scheduler_type (SchedulerType): Enum specifying the scheduler to use.
            optimizer_config (Dict): Dictionary of optimizer hyperparameters (e.g., lr, momentum).
            scheduler_config (Dict): Dictionary of scheduler hyperparameters (e.g., T_max, gamma).
            optimizer_fn (Callable): Function that returns an optimizer when given a model.
            scheduler_fn (Callable): Function that returns a scheduler when given an optimizer.
            local_epochs (int): Number of local epochs to perform.

        Raises:
            ValueError: If optimizer_fn or scheduler_fn is None.
        """
        if optimizer_fn is None or scheduler_fn is None:
            raise ValueError("Both optimizer_fn and scheduler_fn must be provided.")

        self.cid=cid
        self.local_epochs=local_epochs
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        self.optimizer = optimizer_fn(self.model, optimizer_type, optimizer_config)
        self.scheduler = scheduler_fn(self.optimizer, scheduler_type, scheduler_config)

    def get_parameters(self, config):
        """
        Extract model parameters for sending to the server.

        Args:
            config (dict): Not used (placeholder).

        Returns:
            List[np.ndarray]: List of NumPy arrays representing model parameters.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """
        Load model parameters from server.

        Args:
            parameters (List[np.ndarray]): Model parameters as NumPy arrays.
        """
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
      """
      Train the local model for one round.

      Args:
          parameters (List[np.ndarray]): Weights from the global model.
          config (dict): Optional server-sent config (unused here).

      Returns:
          Tuple: Updated weights, number of samples, training metrics.
      """
      print(f"{self.cid}-LOG: Starting local training round")
      self.set_parameters(parameters)
      self.model.train()
      correct, total, loss_total = 0, 0, 0.0

      for epoch in range(self.local_epochs):
          print(f"{self.cid}-LOG: Starting epoch {epoch + 1}/{self.local_epochs}")
          for images, labels in self.trainloader:
              images, labels = images.to(self.device), labels.to(self.device)

              self.optimizer.zero_grad()
              with autocast():
                  outputs = self.model(images)
                  loss = self.criterion(outputs, labels)

              self.scaler.scale(loss).backward()
              self.scaler.step(self.optimizer)
              self.scaler.update()
              self.scheduler.step()

              loss_total += loss.item() * labels.size(0)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

      train_loss = loss_total / total
      train_accuracy = correct / total
      print(f"{self.cid}-LOG: Completed local training - Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")

      return self.get_parameters(config), total, {
          "train_loss": train_loss,
          "train_accuracy": train_accuracy
      }


    def evaluate(self, parameters, config):
        """
        Evaluate the model on the client's validation data.

        Args:
            parameters (List[np.ndarray]): Global model weights.
            config (dict): Optional server config (unused here).

        Returns:
            Tuple: Validation loss, number of samples, validation metrics.
        """
        print(f"{self.cid}-LOG: Starting evaluation")
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss_total += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = loss_total / total
        val_accuracy = correct / total
        print(f"{self.cid}-LOG: Evaluation completed - Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        return val_loss, total, {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

class CIFARFederatedClientModelEditing(fl.client.NumPyClient):
    """
    Federated client for CIFAR-based image classification with Model Editing.
    This client supports:
    - Sparse Fine-Tuning with Masked Gradient Updates.
    - Iterative Pruning to refine masks over local epochs.
    """

    def __init__(
        self,
        cid,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer_config: dict,
        scheduler_config: dict,
        optimizer_type: OptimizerType,
        scheduler_type: SchedulerType,
        optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        local_epochs: int,
        pruning_rounds: int = 4,
        final_sparsity: float = 0.9,
        num_batches: int = 3,
        momentum: float = 0.9,
        weight_decay: float = 0.0005
    ):
        """
        Initialize the federated client with model editing capabilities.

        Args:
            cid: Client id.
            model (nn.Module): PyTorch model instance.
            trainloader (DataLoader): Training dataloader.
            valloader (DataLoader): Validation dataloader.
            device (torch.device): Device to run training on (CPU or CUDA).
            scheduler_config (dict): Configuration parameters for the scheduler.
            optimizer_config (dict): Configuration parameters for the optimizer.
            optimizer_type (OptimizerType): Enum specifying the optimizer to use.
            scheduler_type (SchedulerType): Enum specifying the scheduler to use.
            optimizer_fn (Callable): Function that returns an optimizer when given a model.
            scheduler_fn (Callable): Function that returns a scheduler when given an optimizer.
            local_epochs (int): Number of local epochs to perform.
            pruning_rounds (int): Number of rounds for iterative pruning.
            final_sparsity (float): Final sparsity level (0 < sparsity < 1).
            num_batches (int): Number of batches for Fisher Information scoring.
            momentum (float): Momentum factor for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
        """
        self.cid=cid
        self.local_epochs = local_epochs
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.model = model.to(device)
        self.scaler = GradScaler()
        self.pruner = TaLoSPruner(model, device)

        self.masks = iterative_pruning(
            self.pruner,
            dataloader=trainloader,
            rounds=pruning_rounds,
            final_sparsity=final_sparsity,
            num_batches=num_batches
        )

        for idx, mask in enumerate(self.masks):
            if mask is not None:
                print(f"{self.cid}-LOG: Mask {idx}: Non-zero elements: {torch.sum(mask != 0)} / {mask.numel()}")
            else:
                print(f"{self.cid}-LOG: Mask {idx} is None!")

        self.optimizer = optimizer_fn(
            model=self.model,
            optimizer_type=optimizer_type,
            optimizer_config=optimizer_config,
            masks=self.masks
        )
        
        self.scheduler = scheduler_fn(self.optimizer, scheduler_type, scheduler_config)

    def get_parameters(self, config):
        """Extract model parameters for sending to the server."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
        print("🔍 [Client] get_parameters called.")

    def set_parameters(self, parameters):
        """Load model parameters from server."""
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)
        print("🔍 [Client] set_parameters called.")
    
    def set_global_head(self, global_head_state_dict):
        """
        Apply the global head from the server to the local model.

        Args:
            global_head_state_dict (OrderedDict): The state dictionary of the head.
        """
        print("🔍 Applyed global head")
        head = self.model.head if hasattr(self.model, 'head') else self.model
        head.load_state_dict(global_head_state_dict)


    def set_global_mask(self, global_mask):
        """
        Update the local mask with the global mask received from the server.

        Args:
            global_mask (Dict[str, torch.Tensor]): The global mask for sparsity control.
        """
        print("🔍 Applyed global mask")
        for name, param in self.model.named_parameters():
            if name in global_mask:
                mask = global_mask[name].to(self.device)
                self.masks[name] = mask

    def fit(self, parameters, config):
        """
        Train the local model for one round with Model Editing.
        At the end of local training, it sends back Fisher Information scores
        for global sensitivity calibration.

        Args:
            parameters (List[np.ndarray]): Weights from the global model.
            config (dict): Optional server-sent config.

        Returns:
            Tuple: Updated weights, number of samples, training metrics, sensitivity scores.
        """
        self.set_parameters(parameters)
        self.model.train()
        correct, total, loss_total = 0, 0, 0.0

        if "global_mask" in config:
            self.set_global_mask(config["global_mask"])
        
        if "global_head" in config and config["global_head"] is not None:
            self.set_global_head(config["global_head"])

        # Initialize the Fisher Information dictionary
        fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        for epoch in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Apply the global mask during backpropagation
                self.scaler.scale(loss).backward()
                
                # Mask the gradients accordingly
                for name, param in self.model.named_parameters():
                    if name in self.masks and self.masks[name] is not None:
                        param.grad *= self.masks[name]

                # Update Fisher Information
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_information[name] += (param.grad ** 2) * len(labels)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                loss_total += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Normalize the Fisher Information by total number of samples
        for name in fisher_information:
            fisher_information[name] /= total

        # Convert Fisher Information to NumPy for transmission to the server
        fisher_information_np = {k: v.cpu().numpy() for k, v in fisher_information.items()}

        return self.get_parameters(config), total, {
            "train_loss": loss_total / total,
            "train_accuracy": correct / total,
            "fisher_information": fisher_information_np
        }


    def evaluate(self, parameters, config):
        """
        Evaluate the model on the client's validation data.

        Args:
            parameters (List[np.ndarray]): Global model weights.
            config (dict): Optional server config.

        Returns:
            Tuple: Validation loss, number of samples, validation metrics.
        """
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss_total += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return loss_total / total, total, {
            "val_loss": loss_total / total,
            "val_accuracy": correct / total
        }

