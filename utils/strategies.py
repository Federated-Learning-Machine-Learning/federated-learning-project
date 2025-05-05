from flwr.server.strategy import FedAvg
from torch.utils.tensorboard import SummaryWriter

class FedAvg(FedAvg):
    """
    Federated Averaging (FedAvg) strategy with integrated TensorBoard logging.

    This custom strategy logs both training and evaluation metrics per round:
    - Global and per-client training loss and accuracy
    - Global and per-client validation loss and accuracy
    """

    def __init__(self, writer: SummaryWriter, **kwargs):
        """
        Initialize the strategy with a TensorBoard SummaryWriter.

        Args:
            writer (SummaryWriter): TensorBoard writer for logging.
            **kwargs: Additional keyword arguments passed to the base FedAvg strategy.
        """
        super().__init__(**kwargs)
        self.writer = writer

    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate training results from clients and log metrics to TensorBoard.

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
                if "train_loss" in metrics:
                    self.writer.add_scalar(f"TrainLoss/client_{client_id}", metrics["train_loss"], rnd)
                if "train_accuracy" in metrics:
                    self.writer.add_scalar(f"TrainAccuracy/client_{client_id}", metrics["train_accuracy"], rnd)

        # Perform aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Log aggregated global training metrics
        if aggregated_metrics:
            if "train_loss" in aggregated_metrics:
                self.writer.add_scalar("TrainLoss/global", aggregated_metrics["train_loss"], rnd)
            if "train_accuracy" in aggregated_metrics:
                self.writer.add_scalar("TrainAccuracy/global", aggregated_metrics["train_accuracy"], rnd)

        self.writer.flush()  # Ensure logs are written to disk
        self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """
        Aggregate evaluation results from clients and log metrics to TensorBoard.

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
            self.writer.add_scalar("Loss/global", aggregated_loss, rnd)
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.writer.add_scalar("Accuracy/global", aggregated_metrics["accuracy"], rnd)

        # Log per-client evaluation metrics
        for client_id, eval_res in enumerate(results):
            metrics = eval_res[1].metrics
            if metrics:
                if "val_loss" in metrics:
                    self.writer.add_scalar(f"ValLoss/client_{client_id}", metrics["val_loss"], rnd)
                if "val_accuracy" in metrics:
                    self.writer.add_scalar(f"ValAccuracy/client_{client_id}", metrics["val_accuracy"], rnd)

        self.writer.flush()
        return aggregated_loss, aggregated_metrics
