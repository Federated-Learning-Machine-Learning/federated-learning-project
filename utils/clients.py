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
from editing import SparseSGDM, TaLoSPruner, SparseAdamW
from torch.optim import Optimizer
import numpy as np
import random

class ClientType(Enum):
    STANDARD = "standard"
    FEDPROX = "fedprox"
    TALOS = "talos"
    TALOSPROX = "talosprox"

class OptimizerType(Enum):
    SSGD = "ssgd"
    SADAMW = "sadamw"
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
    batch_size: int,
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
        batch_size (int): Batch Size for data.
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
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size)
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

def build_client_fedprox_fn(
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
    batch_size: int,
    mu: int = 0.1,
    local_epochs: int = 1,
) -> Callable[[Context], Client]:
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
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size)
        print(f"{cid}-LOG: Dataloaders initialized for client {cid}")

        return CIFARFederatedProxClient(
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
            local_epochs=local_epochs,
            mu=mu
        ).to_client()
        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")

    return client_fn

class CIFARFederatedProxClient(CIFARFederatedClient):
    def __init__(
        self,
        cid,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer_type,
        scheduler_type,
        optimizer_config: dict,
        scheduler_config: dict,
        optimizer_fn,
        scheduler_fn,
        local_epochs: int,
        mu: float = 0.1 
    ):
        super().__init__(cid, model, trainloader, valloader, device, optimizer_type, 
                         scheduler_type, optimizer_config, scheduler_config,
                         optimizer_fn, scheduler_fn, local_epochs)
        
        self.mu = mu

    def fit(self, parameters, config):
        """
        Train the local model for one round with FedProx regularization.

        Args:
            parameters (List[np.ndarray]): Weights from the global model.
            config (dict): Optional server-sent config (unused here).

        Returns:
            Tuple: Updated weights, number of samples, training metrics.
        """
        print(f"{self.cid}-LOG: Starting local training round with FedProx (mu={self.mu})")
        
        self.set_parameters(parameters)
        self.model.train()
        correct, total, loss_total = 0, 0, 0.0

        # Save the initial global model weights for Proximal Term computation
        global_weights = [p.clone().detach() for p in self.model.parameters()]

        for epoch in range(self.local_epochs):
            print(f"{self.cid}-LOG: Starting epoch {epoch + 1}/{self.local_epochs}")
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # FedProx proximal term
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_weights):
                        prox_term += (self.mu / 2) * torch.norm(param - global_param) ** 2
                    loss += prox_term

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

def build_client_talos_fn(
    use_iid: bool,
    optimizer_type: OptimizerType,
    optimizer_config: Dict,
    scheduler_type: SchedulerType,
    scheduler_config: Dict,
    iid_partitions: List[Dataset],
    non_iid_partitions: List[Dataset],
    model_fn: Callable[[], nn.Module],
    device: torch_device,
    valset: Dataset,
    batch_size: int,
    talos_config: dict,
    local_epochs: int = 1,
) -> Callable[[Context], Client]:

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_type == "cosine_restart":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
        elif scheduler_type == "multistep":
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_type == "constant":
            return optim.lr_scheduler.ConstantLR(optimizer, **scheduler_config)
        elif scheduler_type == "linear":
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
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size)
        print(f"{cid}-LOG: Dataloaders initialized for client {cid}")

        return CIFARTaLoSClient(
            cid=cid,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            optimizer_type=optimizer_type.value,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            scheduler_type=scheduler_type,
            scheduler_fn=make_scheduler,
            local_epochs=local_epochs,
            talos_config=talos_config
        ).to_client()
        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")

    return client_fn


class CIFARTaLoSClient(fl.client.NumPyClient):
    """
    A Federated Learning client that uses TaLoS Model Editing during local updates.
    """

    def __init__(
        self,
        cid,
        model: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        scheduler_type: str,
        scheduler_config: dict,
        optimizer_type: str,
        optimizer_config: dict,
        talos_config: dict,
        scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        local_epochs: int,

    ):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

        top_layers = None

        if talos_config["mode"] == "head":
            mode="head"
            print(f"⚙️ Initializing {optimizer_type.upper()} for Head Only")
            params_to_train = list(self.model.head.parameters())

        elif talos_config["mode"] == "full":
            mode="full"
            print(f"⚙️ Initializing {optimizer_type.upper()} for Full Model")
            all_params = list(self.model.patch_embed.parameters()) + \
                         list(self.model.norm.parameters()) + \
                         list(self.model.head.parameters())

            for block in self.model.blocks:
                all_params.extend(list(block.parameters()))

            params_to_train = all_params
        elif talos_config["mode"] == "pfededit":
            print(f"⚙️ PFedEdit Mode Activated")

            # === Parameters from Config ===
            self.top_k_layers = talos_config["k"]
            self.stochastic_factor = talos_config["factor"]

            sample = random.random()

            # === Stochastic Sampling ===
            if sample < self.stochastic_factor:
                mode="full"
                print(f"{self.cid}-LOG: Stochastic Sampling Activated, Full Model Training")
                
                # === Select the Full Model Parameters ===
                all_params = list(self.model.patch_embed.parameters()) + \
                            list(self.model.norm.parameters()) + \
                            list(self.model.head.parameters())

                for block in self.model.blocks:
                    all_params.extend(list(block.parameters()))

                params_to_train = all_params
            
            else:
                mode="pfededit"
                print(f"{self.cid}-LOG: PFedEdit Local Layer Selection")
                
                # === Step 1: Generate Candidate Editings ===
                candidate_layers = list(range(len(self.model.blocks)))

                # === Step 2: Evaluate Editings Based on Loss ===
                loss_values = self._evaluate_layer_loss(candidate_layers)

                # === Step 3: Rank Layers Based on Loss ===
                top_layers = sorted(loss_values, key=loss_values.get)[:self.top_k_layers]
                print(f"{self.cid}-LOG: Selected Layers (Min Loss): {top_layers}")
                
                # === Step 4: Collect Parameters for Training ===
                params_to_train = [list(self.model.blocks[i].parameters()) for i in top_layers]
                params_to_train = [p for sublist in params_to_train for p in sublist]

        self.pruner = TaLoSPruner(
            model=self.model,
            device=self.device,
            mode=mode,
            final_sparsity=talos_config["final_sparsity"],
            num_batches=talos_config["num_batches"],
            rounds=talos_config["rounds"],
            layers_to_prune=top_layers,
        )

        mode = talos_config["calibration_mode"]
        
        print(f"{self.cid}-LOG: Starting TaLoS Mask Calibration with mode: {mode}")
        self.pruner.calibrate_masks(self.trainloader, mode)

        # === Selective Layer Editing Logic ===
        
        # === Optimizer Initialization ===
        if optimizer_type == "ssgd":
            self.optimizer = SparseSGDM(
                params=params_to_train,
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"],
                masks=self.pruner.masks
            )

        elif optimizer_type == "sadamw":
            self.optimizer = SparseAdamW(
                params=params_to_train,
                lr=optimizer_config["lr"],
                betas=optimizer_config["betas"],
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
                masks=self.pruner.masks
            )
        else:
            raise ValueError(f"Optimizer type '{optimizer_type}' is not supported.")

        self.scheduler = scheduler_fn(self.optimizer, scheduler_type, scheduler_config)
        
    def get_parameters(self, config):
        """
        Extract model parameters for sending to the server.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """
        Load model parameters from server.
        """
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """
        Train the local model with TaLoS model editing for one round.
        """
        self.round_number = config.get("round_number")
        print(f"{self.cid}-LOG: Starting local training round with TaLoS for round {self.round_number}")
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

    def _evaluate_layer_loss(self, candidate_layers):
        """
        Evaluates the local loss when each candidate layer is personalized.
        This mimics the PFedEdit approach where local layer replacement is tested.

        Args:
            candidate_layers (list): List of layer indices to evaluate.

        Returns:
            dict: A dictionary with layer indices as keys and loss values as values.
        """
        loss_values = {}
        self.model.eval()
        
        # Use a small representative subset of the local data
        representative_subset = list(self.trainloader)[:5]  

        with torch.no_grad():
            for layer_idx in candidate_layers:
                original_state = {n: p.clone() for n, p in self.model.blocks[layer_idx].named_parameters()}
                
                # === Generate candidate editing ===
                # Replace global layer with the locally fine-tuned version
                local_layer = self.model.blocks[layer_idx]
                self.model.blocks[layer_idx].load_state_dict(local_layer.state_dict())

                # === Evaluate the loss ===
                total_loss = 0.0
                for data, target in representative_subset:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, target)
                    total_loss += loss.item()

                # === Store the loss value ===
                loss_values[layer_idx] = total_loss / len(representative_subset)
                
                # === Restore original layer state ===
                self.model.blocks[layer_idx].load_state_dict(original_state)

        return loss_values



def build_client_talos_prox_fn(
    use_iid: bool,
    optimizer_type: OptimizerType,
    optimizer_config: Dict,
    scheduler_type: SchedulerType,
    scheduler_config: Dict,
    iid_partitions: List[Dataset],
    non_iid_partitions: List[Dataset],
    model_fn: Callable[[], nn.Module],
    device: torch_device,
    valset: Dataset,
    batch_size: int,
    talos_config: dict,
    local_epochs: int = 1,
    mu: float = 0.1
) -> Callable[[Context], Client]:

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_type == "cosine_restart":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
        elif scheduler_type == "multistep":
            return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_type == "constant":
            return optim.lr_scheduler.ConstantLR(optimizer, **scheduler_config)
        elif scheduler_type == "linear":
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
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=batch_size)
        print(f"{cid}-LOG: Dataloaders initialized for client {cid}")

        return CIFARTaLoSProxClient(
            cid=cid,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            optimizer_type=optimizer_type.value,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            scheduler_type=scheduler_type,
            scheduler_fn=make_scheduler,
            local_epochs=local_epochs,
            talos_config=talos_config,
            mu=mu,
        ).to_client()
        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")

    return client_fn

class CIFARTaLoSProxClient(CIFARTaLoSClient):
    """
    A Federated Learning client that uses TaLoS Model Editing during local updates.
    Includes FedProx regularization term.
    """

    def __init__(
        self,
        cid,
        model: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device,
        scheduler_type: str,
        scheduler_config: dict,
        optimizer_type: str,
        optimizer_config: dict,
        talos_config: dict,
        scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler],
        local_epochs: int,
        mu: float = 0.1 
    ):
        """
        Initialize the CIFARTaLoSProxClient.
        Inherits all behavior from CIFARTaLoSClient and adds FedProx logic.
        """
        super().__init__(
            cid,
            model,
            trainloader,
            valloader,
            device,
            scheduler_type,
            scheduler_config,
            optimizer_type,
            optimizer_config,
            talos_config,
            scheduler_fn,
            local_epochs
        )
        self.mu = mu
        print(f"{self.cid}-LOG: Initialized with FedProx μ = {self.mu}")

    def fit(self, parameters, config):
          """
          Train the local model for one round with FedProx regularization.

          Args:
              parameters (List[np.ndarray]): Weights from the global model.
              config (dict): Optional server-sent config.

          Returns:
              Tuple: Updated weights, number of samples, training metrics.
          """
          print(f"{self.cid}-LOG: Starting local training round with FedProx (mu={self.mu})")
          
          self.set_parameters(parameters)
          self.model.train()
          correct, total, loss_total = 0, 0, 0.0

          global_weights = {name: p.clone().detach() for name, p in self.model.named_parameters()}

          for epoch in range(self.local_epochs):
              print(f"{self.cid}-LOG: Starting epoch {epoch + 1}/{self.local_epochs}")
              
              for images, labels in self.trainloader:
                  images, labels = images.to(self.device), labels.to(self.device)

                  self.optimizer.zero_grad()
                  with autocast():
                      outputs = self.model(images)
                      loss = self.criterion(outputs, labels)

                      # FedProx proximal term
                      for name, param in self.model.named_parameters():
                          if param.requires_grad and name in global_weights:
                              prox_term = self.mu * 0.5 * torch.norm(param - global_weights[name]) ** 2
                              loss += prox_term

  
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
