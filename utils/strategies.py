from flwr.server.strategy import FedAvg, FedYogi
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, EvaluateRes
from typing import List, Tuple
from wandb_logger import FederatedWandBLogger
import torch
from flwr.common import parameters_to_ndarrays
from enum import Enum

class StrategiesType(Enum):
    FEDAVG = "fedavg"
    YOGI = "yogi"
    METAFEDAVG = "metafedavg"
    
class FedAvgStandard(FedAvg):
    """
    Federated Averaging (FedAvg) strategy with integrated WandB logging.

    This custom strategy logs both training and evaluation metrics per round:
    - Global and per-client training loss and accuracy
    - Global and per-client validation loss and accuracy
    """

    def __init__(self, logger: FederatedWandBLogger, **kwargs):
        """
        Initialize the strategy with a WandB Logger.

        Args:
            logger (FederatedWandBLogger): W&B logger for experiment tracking.
            **kwargs: Additional keyword arguments passed to the base FedAvg strategy.
        """
        super().__init__(**kwargs)
        self.logger = logger

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate training results from clients and log metrics to WandB.

        Args:
            rnd (int): Current training round.
            results (List[Tuple[ClientProxy, FitRes]]): Fit results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated parameters and (optionally) training metrics.
        """
        # Log per-client training metrics
        for client_id, fit_res in enumerate(results):
            metrics = fit_res[1].metrics
            if metrics:
                self.logger.log_client_metrics(client_id, metrics, round_number=rnd)

        # Perform aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Log aggregated global training metrics
        if aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)

        self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """
        Aggregate evaluation results from clients and log metrics to WandB.

        Args:
            rnd (int): Current evaluation round.
            results (List[Tuple[ClientProxy, EvaluateRes]]): Evaluation results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated evaluation loss and metrics.
        """
        # Perform aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Log global evaluation metrics
        if aggregated_loss is not None:
            self.logger.log_global_metrics({"eval_loss": aggregated_loss}, round_number=rnd)
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)

        return aggregated_loss, aggregated_metrics

class FedYogiStandard(FedYogi):
    """
    Federated Yogi (FedYogi) strategy with integrated WandB logging.

    This custom strategy logs both training and evaluation metrics per round:
    - Global and per-client training loss and accuracy
    - Global and per-client validation loss and accuracy
    """

    def __init__(self, logger: FederatedWandBLogger, **kwargs):
        """
        Initialize the strategy with a WandB Logger.

        Args:
            logger (FederatedWandBLogger): W&B logger for experiment tracking.
            **kwargs: Additional keyword arguments passed to the base FedYogi strategy.
        """
        super().__init__(**kwargs)
        self.logger = logger

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[bytes, dict]:
        """
        Aggregate training results from clients and log metrics to WandB.

        Args:
            rnd (int): Current training round.
            results (List[Tuple[ClientProxy, FitRes]]): Fit results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated parameters and (optionally) training metrics.
        """
        print(f"Aggregating fit results for round {rnd}")

        # Log per-client training metrics
        for client_id, (client, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            if metrics:
                self.logger.log_client_metrics(client_id, metrics, round_number=rnd)

        # Perform aggregation using FedYogi
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Log aggregated global training metrics
        if aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)

        self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[float, dict]:
        """
        Aggregate evaluation results from clients and log metrics to WandB.

        Args:
            rnd (int): Current evaluation round.
            results (List[Tuple[ClientProxy, EvaluateRes]]): Evaluation results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated evaluation loss and metrics.
        """
        print(f"Aggregating evaluation results for round {rnd}")

        # Perform aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Log global evaluation metrics
        if aggregated_loss is not None:
            self.logger.log_global_metrics({"eval_loss": aggregated_loss}, round_number=rnd)

        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)

        return aggregated_loss, aggregated_metrics

class MetaFedAvg(FedAvg):
    """
    Meta Federated Averaging (Meta-FedAvg) strategy with integrated WandB logging.

    This strategy introduces a meta-learning step after the standard aggregation:
    - Simulates a local update step for each client post-aggregation
    - Applies the meta-gradient to optimize the global model
    """

    def __init__(self, logger: FederatedWandBLogger, inner_lr: float = 0.01, **kwargs):
        """
        Initialize the strategy with a WandB Logger and inner learning rate for meta updates.

        Args:
            logger (FederatedWandBLogger): W&B logger for experiment tracking.
            inner_lr (float): Learning rate for the meta update step.
            **kwargs: Additional keyword arguments passed to the base FedAvg strategy.
        """
        super().__init__(**kwargs)
        self.logger = logger
        self.inner_lr = inner_lr

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate training results from clients and perform meta-learning update.

        Args:
            rnd (int): Current training round.
            results (List[Tuple[ClientProxy, FitRes]]): Fit results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated parameters and (optionally) training metrics.
        """
        # Log per-client training metrics
        for client_id, fit_res in enumerate(results):
            metrics = fit_res[1].metrics
            if metrics:
                self.logger.log_client_metrics(client_id, metrics, round_number=rnd)

        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        aggregated_tensors = parameters_to_ndarrays(aggregated_parameters)
        meta_gradient = [torch.zeros_like(torch.tensor(p)) for p in aggregated_tensors]

        for client_proxy, fit_res in results:
            # Simulate one local step on the client's model
            local_params = fit_res.parameters
            local_updates = [
                local - global_p
                for local, global_p in zip(local_params, aggregated_parameters)
            ]

            # Meta gradient step: accumulate updates
            for idx, update in enumerate(local_updates):
                meta_gradient[idx] += update

        # Average the meta-gradient
        meta_gradient = [g / len(results) for g in meta_gradient]

        # Apply the meta-gradient to the global model
        self.latest_parameters = [
            global_p - self.inner_lr * g
            for global_p, g in zip(aggregated_parameters, meta_gradient)
        ]

        # Log aggregated global training metrics
        if aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)
            self.logger.log_global_metrics({"meta_loss": aggregated_metrics.get("loss", 0)}, round_number=rnd)

        return self.latest_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """
        Aggregate evaluation results from clients and log metrics to WandB.

        Args:
            rnd (int): Current evaluation round.
            results (List[Tuple[ClientProxy, EvaluateRes]]): Evaluation results from clients.
            failures (List): Clients that failed in this round.

        Returns:
            Tuple: Aggregated evaluation loss and metrics.
        """
        # Perform aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Log global evaluation metrics
        if aggregated_loss is not None:
            self.logger.log_global_metrics({"eval_loss": aggregated_loss}, round_number=rnd)
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.logger.log_global_metrics(aggregated_metrics, round_number=rnd)

        return aggregated_loss, aggregated_metrics
