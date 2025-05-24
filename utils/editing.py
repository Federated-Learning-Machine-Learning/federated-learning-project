import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import numpy as np
from torch.optim import SGD
from torch.optim import AdamW

class SparseAdamW(AdamW):
    """
    Sparse AdamW Optimizer.
    This optimizer selectively updates only parameters specified by masks.

    Args:
        params (iterable): Model parameters to optimize.
        lr (float): Learning rate.
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay (L2 penalty).
        masks (dict): Dictionary of binary masks for parameter updates:
            - ("layer_name", param): Mask Tensor
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, masks=None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.masks = masks
        
        # 📝 Generate a lookup dictionary for fast access
        self.param_to_mask = {}
        
        print("🔍 Mapping parameters to their masks...")

        # Create a fast lookup for parameters to their masks
        for (layer_name, param), mask in self.masks.items():
            self.param_to_mask[param] = mask
        
        print(f"✅ Mapped {len(self.param_to_mask)} parameters to masks.")

    def step(self, closure=None):
        """
        Performs a single optimization step, applying the masks to the gradients
        before updating the parameters.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply mask if available
                if p in self.param_to_mask:
                    mask = self.param_to_mask[p]
                    if mask is not None:
                        p.grad.data.mul_(mask.to(p.grad.device))

        # Proceed with the original AdamW step
        super().step(closure)


class SparseSGDM(SGD):
    """
    Sparse Stochastic Gradient Descent with Momentum (SGDM).
    This optimizer selectively updates only parameters specified by masks.

    Args:
        params (iterable): Model parameters to optimize.
        lr (float): Learning rate.
        momentum (float, optional): Momentum factor. Default: 0.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.
        masks (dict, optional): Dictionary of binary masks for parameter updates:
            - ("layer_name", param): Mask Tensor
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0, masks=None):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.masks = masks
        
        # 📝 Generate a lookup dictionary for fast access
        self.param_to_mask = {}
        
        print("🔍 Mapping parameters to their masks...")

        # Create a fast lookup for parameters to their masks
        for (layer_name, param), mask in self.masks.items():
            self.param_to_mask[param] = mask
        
        print(f"✅ Mapped {len(self.param_to_mask)} parameters to masks.")

    def step(self, closure=None):
        """
        Performs a single optimization step, applying the masks to the gradients
        before updating the parameters.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply mask if available
                if p in self.param_to_mask:
                    mask = self.param_to_mask[p]
                    if mask is not None:
                        p.grad.data.mul_(mask.to(p.grad.device))

        super().step(closure)

class TaLoSPruner:
    """
    TaLoSPruner allows gradient masking and sparse fine-tuning on specific layers:
    - The classifier head
    - Specific Transformer blocks
    """

    def __init__(self, model, device, mode="head",
     final_sparsity=0.9, num_batches=3, rounds=4, layers_to_prune=None):
            """
            Initialize the pruner for specified layers.

            Args:
                model (nn.Module): The vision transformer model.
                device (torch.device): Device to perform pruning on.
                mode (str): "head" to prune only the head, "full" to prune the entire model.
                final_sparsity (float): Final desired sparsity.
                num_batches (int): Number of batches to estimate Fisher Information.
                rounds (int): Rounds of pruning calibration.
                layers_to_prune (list): List of layer names to prune (for PFedEdit).
            """
            self.model = model.to(device)
            self.device = device
            self.final_sparsity = final_sparsity
            self.num_batches = num_batches
            self.rounds = rounds
            self.mode = mode.lower()
            self.masks = {}

            # 📝 Define the layers to prune based on the mode
            if self.mode == "head":
                print("🟢 Pruning will be applied to the classifier head only.")
                self.head = model.head
                self.layers_to_prune = [("head", model.head)]

            elif self.mode == "full":
                print("🟢 Pruning will be applied to the entire model.")
                self.layers_to_prune = []
                
                # Include Patch Embedding
                self.layers_to_prune.append(("patch_embed", model.patch_embed.proj))
                
                # Include Transformer Blocks
                for idx, block in enumerate(model.blocks):
                    self.layers_to_prune.append((f"block_{idx}_attn_qkv", block.attn.qkv))
                    self.layers_to_prune.append((f"block_{idx}_attn_proj", block.attn.proj))
                    self.layers_to_prune.append((f"block_{idx}_mlp_fc1", block.mlp.fc1))
                    self.layers_to_prune.append((f"block_{idx}_mlp_fc2", block.mlp.fc2))
                    self.layers_to_prune.append((f"block_{idx}_norm1", block.norm1))
                    self.layers_to_prune.append((f"block_{idx}_norm2", block.norm2))
                
                # Include Final Layer Norm
                self.layers_to_prune.append(("final_norm", model.norm))
                
                # Include Classifier Head
                self.layers_to_prune.append(("head", model.head))
            elif self.mode == "pfededit":
                print(f"🟢 Pruning will be applied to the following layers: {layers_to_prune}")
                self.layers_to_prune = [(f"block_{i}", self.model.blocks[i]) for i in layers_to_prune]
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'head' or 'full'.")

    def fisher_information(self, dataloader):
            """
            Computes the Fisher Information matrix for parameter sensitivity estimation.
            """
            fisher_scores = {}

            # Initialize Fisher scores for each layer
            for name, layer in self.layers_to_prune:
                for p in layer.parameters():
                    fisher_scores[(name, p)] = torch.zeros_like(p, device=self.device)

            print(f"📝 Calculating Fisher Information on {self.num_batches} batches...")

            self.model.train()

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= self.num_batches:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.model.zero_grad()

                # Forward pass through the model
                inputs = inputs.to(self.device).float()
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)

                # Backward pass
                loss.backward()

                # Accumulate the Fisher Information scores
                for (name, layer) in self.layers_to_prune:
                    for p in layer.parameters():
                        if p.grad is not None:
                            fisher_scores[(name, p)] += (p.grad.data ** 2) * inputs.size(0)

            # Normalize the scores
            for key in fisher_scores:
                fisher_scores[key] /= self.num_batches

            print("✅ Fisher Information Computation Completed.")
            return fisher_scores

    def _generate_masks(self, all_scores, strategy):
        """
        Helper method to generate binary masks based on sensitivity scores.
        """
        masks = {}
        for (layer_name, param), score in all_scores:
            if strategy == "least_sensitive":
                threshold = torch.quantile(score, 1 - self.final_sparsity)
                mask = (score <= threshold).float().reshape(param.shape).to(self.device)
            elif strategy == "most_sensitive":
                threshold = torch.quantile(score, self.final_sparsity)
                mask = (score >= threshold).float().reshape(param.shape).to(self.device)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            masks[(layer_name, param)] = mask
        return masks

        
    def calibrate_masks(self, dataloader, strategy="least_sensitive"):
        """
        Multi-round mask calibration to refine parameter selection.
        Pruning is applied to:
        - The classifier head if mode is "head"
        - The entire model if mode is "full"

        Args:
            dataloader: DataLoader object for sampling batches.
            strategy (str): Strategy for pruning, either "least_sensitive" or "most_sensitive".
        """
        print(f"🔎 Starting multi-round calibration for mode '{self.mode}'.")
        
        for round_num in range(self.rounds):
            print(f"🌀 Calibration Round {round_num + 1}/{self.rounds}")
            
            # Step 1: Compute Fisher Information for all components
            fisher_scores = self.fisher_information(dataloader)

            # Step 2: Flatten and sort by sensitivity
            def _flatten_and_sort(fisher_scores):
                """
                Helper to flatten and sort Fisher scores
                """
                all_scores = []
                for (layer_name, param), score in fisher_scores.items():
                    all_scores.append(((layer_name, param), score.flatten()))
                all_scores.sort(key=lambda x: x[1].mean(), reverse=True)
                return all_scores

            # Step 3: Generate Masks
            all_scores = _flatten_and_sort(fisher_scores)
            self.masks = self._generate_masks(all_scores, strategy)

        print("✅ Mask Calibration Completed!")
