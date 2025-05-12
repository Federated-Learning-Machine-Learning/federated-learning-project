import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import numpy as np
from torch.optim import SGD


class SparseSGDM(SGD):
    """
    Sparse Stochastic Gradient Descent with Momentum (SGDM).
    This optimizer selectively updates only parameters specified by masks.

    Args:
        params (iterable): Model parameters to optimize.
        lr (float): Learning rate.
        momentum (float, optional): Momentum factor. Default: 0.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.
        masks (list, optional): List of binary masks for parameter updates. Default: None.
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0, masks=None):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.masks = masks

    def step(self, closure=None):
        """
        Performs a single optimization step, applying the masks to the gradients
        before updating the parameters.
        """
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                # Apply mask if available
                if self.masks is not None and self.masks[i] is not None:
                    p.grad.data.mul_(self.masks[i].to(p.grad.device))
        super().step(closure)

        
class TaLoSPruner:
    def __init__(self, model, device, final_sparsity=0.9, num_batches=3, rounds=4):
        self.model = model
        self.device = device
        self.final_sparsity = final_sparsity
        self.num_batches = num_batches
        self.rounds = rounds
        self.masks = None

        # Identify the head of the model (DINO ViT for example)
        if hasattr(model, 'head'):
            print("🟢 Found model head. Pruning will be applied to the head only.")
            self.head = model.head
        else:
            print("⚠️ No specific 'head' found. Applying pruning to the entire model.")
            self.head = model

    def fisher_information(self, dataloader):
        """
        Computes the Fisher Information matrix for parameter sensitivity estimation.
        Only parameters in the `head` are considered.
        """
        fisher_scores = {p: torch.zeros_like(p, device=self.device) for p in self.head.parameters()}
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= self.num_batches:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()

            # Forward pass through the head
            with autocast():
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)

            loss.backward()

            # Accumulate the Fisher Information scores
            for p in self.head.parameters():
                if p.grad is not None:
                    fisher_scores[p] += (p.grad.data ** 2) * inputs.size(0)

        # Normalize the scores
        for p in fisher_scores:
            fisher_scores[p] /= self.num_batches

        return fisher_scores

    def calibrate_masks(self, dataloader, strategy="least_sensitive"):
        """
        Multi-round mask calibration to refine parameter selection.
        Only parameters in the `head` are considered for pruning.
        """
        print(f"🔎 Starting multi-round calibration for the model head only.")
        for round_num in range(self.rounds):
            print(f"🌀 Calibration Round {round_num + 1}/{self.rounds}")
            fisher_scores = self.fisher_information(dataloader)

            # Flatten and sort by sensitivity
            all_scores = []
            for param, score in fisher_scores.items():
                all_scores.append((score.flatten(), param))

            all_scores.sort(key=lambda x: x[0].mean(), reverse=True)

            # Update sparsity for this round
            current_sparsity = self.final_sparsity * (round_num + 1) / self.rounds
            total_params = sum(p.numel() for p in self.head.parameters())
            num_keep = int((1 - current_sparsity) * total_params)

            # Generate masks
            self.masks = []
            strategy = strategy.lower()
            if strategy == "least_sensitive":
                for score, param in all_scores:
                    threshold = torch.quantile(score, 1 - current_sparsity) 
                    mask = (score <= threshold).float().reshape(param.shape).to(self.device) #keeps weights with score lower than threshold
                    self.masks.append(mask)
            if strategy == "most_sensitive":
                for score, param in all_scores:
                    threshold = torch.quantile(score, current_sparsity)
                    mask = (score >= threshold).float().reshape(param.shape).to(self.device)
                    self.masks.append(mask)

            # Apply the new masks
            #self.apply_masks()
            print(f"✅ Mask updated with {current_sparsity * 100:.2f}% sparsity")

    """def apply_masks(self):
        
        #Apply the binary masks to the head parameters only.
        
        with torch.no_grad():
            for mask, param in zip(self.masks, self.head.parameters()):
                param.data *= mask"""
