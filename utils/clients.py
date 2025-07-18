"""
clients.py contains the implementation of Flower clients for federated learning experiments.
- CIFARFederatedClient: A standard client for CIFAR-based image classification.
- CIFARFederatedProxClient: A client that implements FedProx regularization.
- CIFARTaLoSClient: A client that uses TaLoS model editing during local updates. 
- CIFARTalosProxClient: A client that combines TaLoS with FedProx regularization.
- PFedEditClient: A client that implements PFedEdit for selective layer training.
- PFedEditProxClient: A client that combines PFedEdit with FedProx
- TalosPFedEditClient: A client that combines TaLoS with PFedEdit for selective layer training.
"""


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
    PFEDEDIT = "pfededit"
    PFEDEDITPROX = "pfededitprox"
    TALOSPFEDEDIT = "talospfededit"

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

    """
    A Flower client that implements FedProx regularization during local training.
    args: 
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
        mu (float): FedProx regularization parameter.
    
    """
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
        mu: float = 0.1,
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
    args:
        cid: Client id.
        model (nn.Module): PyTorch model instance.
        trainloader (DataLoader): Training dataloader.
        valloader (DataLoader): Validation dataloader.
        device (torch.device): Device to run training on (CPU or CUDA).
        optimizer_type (str): Type of optimizer to use (e.g., 'ssgd', 'sadamw').
        scheduler_type (str): Type of scheduler to use (e.g., 'cosine', 'step').
        optimizer_config (dict): Configuration for the optimizer.
        scheduler_config (dict): Configuration for the scheduler.
        scheduler_fn (Callable): Function to create the scheduler.
        local_epochs (int): Number of local epochs to perform.
        talos_config (dict): Configuration for TaLoS model editing.
    
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
    A Flower client that implements FedProx regularization during local training.
    args:  
        cid: Client id.
        model (nn.Module): PyTorch model instance.
        trainloader (DataLoader): Training dataloader.
        valloader (DataLoader): Validation dataloader.
        device (torch.device): Device to run training on (CPU or CUDA).
        optimizer_type (str): Type of optimizer to use (e.g., 'ssgd', 'sadamw').
        scheduler_type (str): Type of scheduler to use (e.g., 'cosine', 'step').
        optimizer_config (dict): Configuration for the optimizer.
        scheduler_config (dict): Configuration for the scheduler.
        scheduler_fn (Callable): Function to create the scheduler.
        local_epochs (int): Number of local epochs to perform.
        mu (float): FedProx regularization parameter.
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


def build_client_fn_pfededit(
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
    local_epochs: int = 1,
    top_k_layers: int = 2,
    max_batches: int = 4,
    rounds_stochastic: int = 8
) -> Callable[[Context], Client]:
    """
    Build a client function for PFedEdit.
    """

    def make_optimizer(params: List[nn.Parameter], optimizer_type: str, optimizer_config: dict) -> optim.Optimizer:
        if optimizer_type == OptimizerType.SGD:
            return optim.SGD(params, **optimizer_config)
        elif optimizer_type == OptimizerType.ADAM:
            return optim.Adam(params, **optimizer_config)
        elif optimizer_type == OptimizerType.ADAMW:
            return optim.AdamW(params, **optimizer_config)
        elif optimizer_type == OptimizerType.RMSPROP:
            return optim.RMSprop(params, **optimizer_config)
        elif optimizer_type == OptimizerType.ADAGRAD:
            return optim.Adagrad(params, **optimizer_config)
        else:
            raise ValueError(f"Optimizer type {optimizer_type} not supported.")


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
        
        
    def client_fn(context: Context) -> Client:
        """
        Construct a client instance based on the context provided by Flower.
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

        # Create a NumPyClient instance
        return PFedEditClient(
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
            top_k_layers=top_k_layers,
            max_batches=max_batches,
            rounds_stochastic=rounds_stochastic
        ).to_client()
        
        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")
    
    return client_fn
    

class PFedEditClient(fl.client.NumPyClient):
    """
    A Flower client that implements the PFedEdit strategy for local model training.
    args:
        cid (int): Client ID.
        model (nn.Module): Local model to be trained.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on (CPU or GPU).
        optimizer_config (Dict): Configuration for the optimizer.
        optimizer_type (str): Type of optimizer to use (e.g., 'sgd', 'adam').
        scheduler_config (Dict): Configuration for the learning rate scheduler.
        scheduler_type (str): Type of scheduler to use (e.g., 'cosine', 'step').
        optimizer_fn (Callable): Function to create the optimizer.
        scheduler_fn (Callable): Function to create the scheduler.
        local_epochs (int): Number of local training epochs.
        top_k_layers (int): Number of top layers to train based on PFedEdit.
        max_batches (int): Maximum number of batches to use for training.
        rounds_stochastic (int): Number of rounds 
    """
    def __init__(
        self,
        cid: int,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        optimizer_config: Dict,
        optimizer_type: str,
        scheduler_config: Dict,
        scheduler_type: str,
        optimizer_fn,
        scheduler_fn,
        local_epochs: int = 1,
        top_k_layers: int = 2,
        max_batches: int = 4,
        rounds_stochastic: int = 8
    ):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.optimizer_config = optimizer_config
        self.optimizer_type = optimizer_type
        self.scheduler_config = scheduler_config
        self.scheduler_type = scheduler_type
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.local_epochs = local_epochs
        self.top_k_layers = top_k_layers
        self.stochastic_factor = 1.0
        self.max_batches = max_batches,
        self.rounds_stochastic = rounds_stochastic
    
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
        """Set the local model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
    

    def fit(self, parameters, config):
        """Train the model using the provided parameters."""
        self.set_parameters(parameters)
        self.model.train()
        # Retrieve current rouuf nd and total rounds from config
        current_round = config.get("current_round", 0)
        total_rounds = self.rounds_stochastic  # fallback
        print("Current Round:", current_round)
        print("Total Rounds:", total_rounds)

        initial_stochastic = 1.0
        progress = (current_round - 1) / max(total_rounds - 1, 1)
        current_stochastic = initial_stochastic * (1.0 - progress)
        current_stochastic = max(current_stochastic, 0.0)

        print(f"{self.cid}-LOG: Round {current_round} | Dynamic Stochastic Factor: {current_stochastic:.4f}")

        params_to_train = self.select_params_to_train(current_stochastic)

        optimizer = self.optimizer_fn(params_to_train, self.optimizer_type, self.optimizer_config)
        scheduler = self.scheduler_fn(optimizer, self.scheduler_type, self.scheduler_config)


        loss_total = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()

                loss_total += loss.item() * images.size(0)  # somma pesata per batch size
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scheduler.step()

        train_loss = loss_total / total
        train_accuracy = correct / total
        print(f"{self.cid}-LOG: Completed local training - Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")

        return self.get_parameters({}), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        }


    def evaluate(self, parameters, config):
        """Evaluate the model using the provided parameters."""
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += nn.CrossEntropyLoss()(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"{self.cid}-LOG: Evaluation completed - Evaluation Loss: {loss / total:.4f} | Evaluation Accuracy: {correct / total:.4f}")
        return loss / total, total, {"val_accuracy": correct / total}

    def select_params_to_train(self, stochastic_factor: float):
        """Select which parameters to train based on the PFedEdit strategy."""
        sample = random.random()

        if sample < stochastic_factor:
            print(f"{self.cid}-LOG: Stochastic Sampling Activated (stochastic_factor={stochastic_factor:.4f}), Full Model Training")
            all_params = list(self.model.parameters())
        else:
            print(f"{self.cid}-LOG: PFedEdit Local Layer Selection (stochastic_factor={stochastic_factor:.4f})")
            candidate_layers = list(range(len(self.model.blocks)))
            loss_values = self._evaluate_layer_loss(candidate_layers, max_batches=self.max_batches)
            top_layers = sorted(loss_values, key=loss_values.get)[:self.top_k_layers]
            print(f"{self.cid}-LOG: Selected Layers (Min Loss): {top_layers}")
            all_params = []
            for idx in top_layers:
                all_params += list(self.model.blocks[idx].parameters())
            all_params += list(self.model.head.parameters())
            all_params += list(self.model.norm.parameters())

        return all_params

    def _evaluate_layer_loss(self, candidate_layers, max_batches=4):
        """Evaluate the loss for each layer."""
        loss_values = {}
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        print("Max Batches:", max_batches)

        with torch.no_grad():
            for idx in candidate_layers:
                layer_loss = 0.0
                batch_count = 0
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self._forward_with_mask(idx, images)
                    layer_loss += criterion(output, labels).item()
                    batch_count += 1
                    if batch_count >= max_batches:
                        break
                if batch_count == 0:
                    loss_values[idx] = float("inf")
                else:
                    loss_values[idx] = layer_loss / batch_count

        return loss_values


    def _forward_with_mask(self, active_layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with only the active layer enabled."""
        x = self.model.patch_embed(x)
        for i, blk in enumerate(self.model.blocks):
            if i == active_layer_idx:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])
        return x
    

def build_client_fn_pfededitprox(
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
    mu: float = 0.1,
    rounds_stochastic: int = 8
) -> Callable[[Context], Client]:

    """
    Build a client function for PFedEdit with FedProx.
    """

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
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

        # Create a NumPyClient instance
        return PFedEditProxClient(
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
            rounds_stochastic=rounds_stochastic
        ).to_client()
    
    return client_fn


class PFedEditProxClient(PFedEditClient):
    """
    A Federated Learning client that uses PFedEdit with FedProx during local updates.

    args:
        cid (int): Client ID.
        model (nn.Module): Local model.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on.
        optimizer_config (Dict): Configuration for the optimizer.
        optimizer_type (str): Type of optimizer to use.
        scheduler_config (Dict): Configuration for the learning rate scheduler.
        scheduler_type (str): Type of scheduler to use.
        optimizer_fn: Function to create the optimizer.
        scheduler_fn: Function to create the scheduler.
        local_epochs (int): Number of local epochs to train.
        mu (float): FedProx regularization parameter.
        rounds_stochastic (int): Number of stochastic rounds for dynamic sampling.

    """

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        optimizer_config: Dict,
        optimizer_type: str,
        scheduler_config: Dict,
        scheduler_type: str,
        optimizer_fn,
        scheduler_fn,
        local_epochs: int = 1,
        mu: float = 0.1,
        rounds_stochastic: int = 8
    ):
        super().__init__(
            cid=cid,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            optimizer_config=optimizer_config,
            optimizer_type=optimizer_type,
            scheduler_config=scheduler_config,
            scheduler_type=scheduler_type,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            local_epochs=local_epochs
        )
        self.mu = mu
        print(f"{self.cid}-LOG: Initialized with FedProx μ = {self.mu}")
    
    def fit(self, parameters, config):
        """
        Train the local model with PFedEdit and FedProx for one round.
        """
        self.set_parameters(parameters)
        self.model.train()
        correct, total, loss_total = 0, 0, 0.0

        # Retrieve current round and total rounds from config
        current_round = config.get("current_round", 0)
        total_rounds = self.rounds_stochastic  # fallback
        print("Current Round:", current_round)
        print("Total Rounds:", total_rounds)
        initial_stochastic = 1.0
        progress = (current_round - 1) / max(total_rounds - 1, 1)
        current_stochastic = initial_stochastic * (1.0 - progress)
        current_stochastic = max(current_stochastic, 0.0)
        print(f"{self.cid}-LOG: Round {current_round} | Dynamic Stochastic Factor: {current_stochastic:.4f}")
        params_to_train = self.select_params_to_train(current_stochastic)
        optimizer = self.optimizer_fn(params_to_train, self.optimizer_type, self.optimizer_config)
        scheduler = self.scheduler_fn(optimizer, self.scheduler_type, self.scheduler_config)
        loss_total = 0.0
        correct = 0
        total = 0
        global_weights = {name: p.clone().detach() for name, p in self.model.named_parameters()}
        for epoch in range(self.local_epochs):
            print(f"{self.cid}-LOG: Starting epoch {epoch + 1}/{self.local_epochs}")
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output, labels)

                # FedProx proximal term
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in global_weights:
                        prox_term = self.mu * 0.5 * torch.norm(param - global_weights[name]) ** 2
                        loss += prox_term

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_total += loss.item() * images.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_loss = loss_total / total
        train_accuracy = correct / total
        print(f"{self.cid}-LOG: Completed local training - Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        }

def build_client_fn_talos_pfededit(
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
    pfededit_config: dict,
    local_epochs: int = 1,
    mu: float = 0.1,
    rounds_stochastic: int = 8,
    deterministic_round: int = 4,
    all_rounds_scheduling: bool = True,
    reverse_mode: bool = False
) -> Callable[[Context], Client]:
    """
    Build a client function for TaLoS with PFedEdit.
    """

    def make_scheduler(optimizer: optim.Optimizer, scheduler_type: str, scheduler_config: dict) -> optim.lr_scheduler._LRScheduler:
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

        # Create a NumPyClient instance
        return TaLoSPFedEditClient(
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
            pfededit_config=pfededit_config,
            mu=mu,
            rounds_stochastic = rounds_stochastic,
            deterministic_round=deterministic_round,
            all_rounds_scheduling=all_rounds_scheduling,
            reverse_mode=reverse_mode
        ).to_client()

        print(f"{cid}-LOG: Client {cid} fully initialized and ready for training")
    return client_fn

        
class TaLoSPFedEditClient(fl.client.NumPyClient):
    """
    A Federated Learning client that uses TaLoS with PFedEdit during local updates.

    args:
        cid (int): Client ID.
        model (nn.Module): Local model.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on.
        optimizer_config (Dict): Configuration for the optimizer.
        optimizer_type (str): Type of optimizer to use.
        scheduler_config (Dict): Configuration for the learning rate scheduler.
        scheduler_type (str): Type of learning rate scheduler to use.
        scheduler_fn: Function to create the scheduler.
        talos_config (dict): Configuration for TaLoS.
        pfededit_config (dict): Configuration for PFedEdit.
        local_epochs (int, optional): Number of local epochs. Defaults to 1.
        mu (float, optional): FedProx regularization parameter. Defaults to 0.1.
        rounds_stochastic (int, optional): Number of rounds of federated learning process. Defaults to 8.
        deterministic_round (int, optional): Round for deterministic scheduling. Defaults to 4.
        all_rounds_scheduling (bool, optional): Whether to use all rounds scheduling. Defaults to True.
        reverse_mode (bool, optional): Whether to use reverse mode for scheduling. Defaults to False.

    returns:
        fl.client.Client: A fully initialized Flower client with TaLoS and PFedEdit.
    
    """

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        optimizer_config: Dict,
        optimizer_type: str,
        scheduler_config: Dict,
        scheduler_type: str,
        scheduler_fn,               
        talos_config: dict,
        pfededit_config: dict,
        local_epochs: int = 1,
        mu: float = 0.1,
        rounds_stochastic: int = 8,
        deterministic_round: int = 4,
        all_rounds_scheduling: bool = True,
        reverse_mode: bool = False
    ):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.optimizer_config = optimizer_config
        self.optimizer_type = optimizer_type
        self.scheduler_config = scheduler_config
        self.scheduler_type = scheduler_type
        self.scheduler_fn = scheduler_fn
        self.local_epochs = local_epochs
        self.mu = mu
        self.pfededit_config = pfededit_config
        self.talos_config = talos_config
        self.top_k_layers = pfededit_config["top_k_layers"]
        self.max_batches = pfededit_config["max_batches"]
        self.rounds_stochastic = rounds_stochastic
        self.deterministic_round = deterministic_round
        self.all_rounds_scheduling = all_rounds_scheduling
        self.reverse_mode = reverse_mode

        print(f"{self.cid}-LOG: Initialized with FedProx μ = {self.mu}")
    
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
        """Set the local model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train the local model with TaLoS and PFedEdit for one round.
        """
        self.set_parameters(parameters)
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.train()
        correct, total, loss_total = 0, 0, 0.0
        initial_stochastic = 1.0

        if self.all_rounds_scheduling:
            # Retrieve current round and total rounds from config
            current_round = config.get("current_round", 0)
            total_rounds = self.rounds_stochastic
            print("Current Round:", current_round)
            print("Total Rounds:", total_rounds)

            #Scheduling the stochastic factor
            if self.reverse_mode:
                # Reverse mode for scheduling
                print(f"{self.cid}-LOG: Using reverse mode for scheduling")
                progress = (current_round - 1) / max(total_rounds - 1, 1)
                current_stochastic = min(max(progress, 0.0), 1.0)
            else:
                # Normal mode for scheduling
                print(f"{self.cid}-LOG: Using normal mode for scheduling")
                progress = (current_round - 1) / max(total_rounds - 1, 1)
                current_stochastic = initial_stochastic * (1.0 - progress)
                current_stochastic = max(current_stochastic, 0.0)
        else:
            # Use deterministic round for scheduling
            current_round = config.get("current_round", 0)
            total_rounds = self.deterministic_round
            print("Current Round:", current_round)
            print("Total Rounds:", total_rounds)
            if self.reverse_mode:
                progress = (current_round - 1) / max(total_rounds - 1, 1)
                current_stochastic = min(max(progress, 0.0), 1.0)
            else:
                print(f"{self.cid}-LOG: Using normal mode for scheduling")
                progress = (current_round - 1) / max(total_rounds - 1, 1)
                current_stochastic = initial_stochastic * (1.0 - progress)
                current_stochastic = max(current_stochastic, 0.0)

        print(f"{self.cid}-LOG: Round {current_round} | Dynamic Stochastic Factor: {current_stochastic:.4f}")

        params_to_train, top_layers, mode = self.select_params_to_train(current_stochastic)


        #if mode is not pfededit, use Talos (i.e. apply the pruning) along all the layers
        if mode != "pfededit": 
            self.pruner = TaLoSPruner(
                model=self.model,
                device=self.device,
                mode=mode,
                final_sparsity=self.talos_config["final_sparsity"],
                num_batches=self.talos_config["num_batches"],
                rounds=self.talos_config["rounds"],
            )
        else: #the mode is pfededit, use Talos (i.e. apply the pruning) only on the top selected k layers
            self.pruner = TaLoSPruner(
                model=self.model,
                device=self.device,
                mode=mode,
                final_sparsity=self.talos_config["final_sparsity"],
                num_batches=self.talos_config["num_batches"],
                rounds=self.talos_config["rounds"],
                layers_to_prune=top_layers,
            )

        print(f"{self.cid}-LOG: Starting TaLoS calibration in mode: {mode}")

        self.pruner.calibrate_masks(self.trainloader, strategy=self.talos_config["calibration_mode"]) # calibrate the masks based on the selected mode


        # === Optimizer Initialization ===
        if self.optimizer_type == "ssgd":
            optimizer = SparseSGDM(
                params=params_to_train,
                lr=self.optimizer_config["lr"],
                momentum=self.optimizer_config["momentum"],
                weight_decay=self.optimizer_config["weight_decay"],
                masks=self.pruner.masks
            )
            print("{}-LOG: Using SparseSGDM optimizer".format(self.cid))

        elif self.optimizer_type == "sadamw":
            optimizer = SparseAdamW(
                params=params_to_train,
                lr=self.optimizer_config["lr"],
                betas=self.optimizer_config["betas"],
                eps=self.optimizer_config["eps"],
                weight_decay=self.optimizer_config["weight_decay"],
                masks=self.pruner.masks
            )
            print("{}-LOG: Using SparseAdamW optimizer".format(self.cid))
        else:
            raise ValueError(f"Optimizer type is not supported.")

        scheduler = self.scheduler_fn(optimizer, self.scheduler_type, self.scheduler_config)

        loss_total = 0.0
        correct = 0
        total = 0
        global_weights = {name: p.clone().detach() for name, p in self.model.named_parameters()}

        for epoch in range(self.local_epochs):
            print(f"{self.cid}-LOG: Starting epoch {epoch + 1}/{self.local_epochs}")
            for images, labels in self.trainloader:
                images, labels = images.to(self.device).float(), labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output, labels)

                # FedProx proximal term
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in global_weights:
                        prox_term = self.mu * 0.5 * torch.norm(param - global_weights[name]) ** 2
                        loss += prox_term

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_total += loss.item() * images.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = loss_total / total
        train_accuracy = correct / total
        print(f"{self.cid}-LOG: Completed local training - Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
        return self.get_parameters(config), len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        }
    
    def select_params_to_train(self, stochastic_factor: float):
        """
        Select which parameters to train based on the TaLoS strategy.
        """
        sample = random.random()
        print(f"{self.cid}-LOG: Random Sample: {sample:.4f}")

        if sample < stochastic_factor:
            print(f"{self.cid}-LOG: Stochastic Sampling Activated (stochastic_factor={stochastic_factor:.4f}), Full Model Training")
            all_params = list(self.model.parameters())
            return all_params, None, self.talos_config["mode"]
        else:
            print(f"{self.cid}-LOG: TaLoS Local Layer Selection (stochastic_factor={stochastic_factor:.4f})")
            candidate_layers = list(range(len(self.model.blocks)))
            loss_values = self._evaluate_layer_loss(candidate_layers, max_batches=self.max_batches)
            top_layers = sorted(loss_values, key=loss_values.get)[:self.top_k_layers]
            print(f"{self.cid}-LOG: Selected Layers (Min Loss): {top_layers}")
            all_params = []
            for idx in top_layers:
                all_params += list(self.model.blocks[idx].parameters())
            all_params += list(self.model.head.parameters())
            all_params += list(self.model.norm.parameters())
            return all_params, top_layers, "pfededit"
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on the client's validation data.
        """
        print(f"{self.cid}-LOG: Starting evaluation")
        self.set_parameters(parameters)
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)

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

    def _evaluate_layer_loss(self, candidate_layers, max_batches=4):
        """
        Evaluate the loss for each layer.
        """
        loss_values = {}
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        print("Max Batches:", max_batches)

        with torch.no_grad():
            for idx in candidate_layers:
                layer_loss = 0.0
                batch_count = 0
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self._forward_with_mask(idx, images)
                    layer_loss += criterion(output, labels).item()
                    batch_count += 1
                    if batch_count >= max_batches:
                        break
                if batch_count == 0:
                    loss_values[idx] = float("inf")
                else:
                    loss_values[idx] = layer_loss / batch_count

        return loss_values
    
    def _forward_with_mask(self, active_layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with only the active layer enabled.
        """
        x = x.to(self.device)
        x = self.model.patch_embed(x)
        for i, blk in enumerate(self.model.blocks):
            if i == active_layer_idx:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])
        return x
    
