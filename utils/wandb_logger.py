import wandb
import torch

class WandBLogger:
    def __init__(self, project_name, run_name, config=None):
        """
        Initialize the WandB Logger.
        
        Args:
            project_name (str): The name of the project in WandB.
            run_name (str): The specific name for this run.
            config (dict): Optional configuration parameters to log.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        wandb.init(project=self.project_name, name=self.run_name, entity='polito-fl')
        wandb.config.update(self.config)
    
    def log_metrics(self, metrics, step=None):
        """
        Log a dictionary of metrics.
        
        Args:
            metrics (dict): A dictionary where keys are metric names and values are metric values.
            step (int): Optional step number for the metrics.
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_model(self, model, path="model.pth"):
        """
        Save the model checkpoint and log it to WandB.
        
        Args:
            model (torch.nn.Module): The PyTorch model to save.
            path (str): The path where the model will be saved.
        """
        torch.save(model.state_dict(), path)
        wandb.save(path)
    
    def log_artifact(self, path, name="artifact"):
        """
        Log an artifact (e.g., dataset, file).
        
        Args:
            path (str): The local path of the artifact.
            name (str): The name under which it will be saved in WandB.
        """
        wandb.log_artifact(path, name=name)
    
    def finish(self):
        """
        Finish the WandB run.
        """
        wandb.finish()

class FederatedWandBLogger:
    def __init__(self, project_name, run_name, global_config=None):
        """
        Initialize the WandB Logger for Federated Learning.

        Args:
            project_name (str): The name of the project in WandB.
            run_name (str): The specific name for this run.
            global_config (dict): Optional configuration parameters to log globally.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.global_config = global_config or {}

        # Initialize the WandB run
        wandb.init(project=self.project_name, name=self.run_name, entity='polito-fl')
        wandb.config.update(self.global_config)

    def log_global_metrics(self, metrics, round_number):
        """
        Log global training and evaluation metrics.

        Args:
            metrics (dict): Dictionary of metrics to log.
            round_number (int): Current federated round.
        """
        wandb.log({f"global/{key}": value for key, value in metrics.items()}, step=round_number)

    def log_client_metrics(self, client_id, metrics, round_number):
        """
        Log per-client training and evaluation metrics.

        Args:
            client_id (int): ID of the client.
            metrics (dict): Dictionary of metrics to log.
            round_number (int): Current federated round.
        """
        wandb.log({f"client_{client_id}/{key}": value for key, value in metrics.items()}, step=round_number)

    def log_model(self, model, path="model.pth"):
        """
        Save the global model checkpoint and log it to WandB.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            path (str): The path where the model will be saved.
        """
        torch.save(model.state_dict(), path)
        wandb.save(path)

    def finish(self):
        """
        Finish the WandB run.
        """
        wandb.finish()
    ""
