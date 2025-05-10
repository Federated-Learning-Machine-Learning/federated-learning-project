import torch
import torch.nn as nn
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
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scores = {}

        # Verificare la struttura del modello
        if hasattr(model, 'head'):
            self.head = model.head
        else:
            print("Attenzione: il modello non ha un attributo 'head'. Utilizzo l'intero modello.")
            self.head = model

    def score(self, dataloader, num_batches=1):
        self.model.eval()

        # Inizializza dizionario dei punteggi
        for p in self.head.parameters():
            self.scores[p] = torch.zeros_like(p, device=self.device)

        total_samples = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            x, y = x.to(self.device), y.to(self.device)

            self.model.zero_grad()

            # Forward pass con gestione di diversi tipi di modelli
            if hasattr(self.model, 'forward_features'):
                with torch.no_grad():
                    features = self.model.forward_features(x)

                # Gestisci diverse strutture di output
                if features.ndim == 3:  # [B, seq_len, hidden_dim]
                    cls_token = features[:, 0, :]  # Prendi solo il CLS token
                else:
                    cls_token = features  # Già nella forma corretta

                # Forward pass solo sulla testa
                output = self.head(cls_token)
            else:
                # Se il modello non ha forward_features, usa il forward normale
                self.model.zero_grad()
                output = self.model(x)

            # Adatta l'output se necessario
            if output.ndim == 3:
                output = output.squeeze(1)

            # Calcolo della loss
            loss = nn.CrossEntropyLoss()(output, y)

            # Backward per calcolare i gradienti
            loss.backward()

            # Accumula i punteggi (Fisher diagonale approssimato)
            for p in self.head.parameters():
                if p.grad is not None:
                    self.scores[p] += (p.grad.data ** 2) * x.size(0)

            total_samples += x.size(0)

        # Normalizza i punteggi
        for p in self.head.parameters():
            if p in self.scores and total_samples > 0:
                self.scores[p] /= total_samples


    def generate_masks(self, sparsity=0.5):
        """
        Generate binary masks based on the Fisher Information scores.

        Args:
            sparsity (float): Fraction of parameters to prune (0 < sparsity < 1).

        Returns:
            list: Binary masks for each parameter in the head.
        """
        masks = []

        for p in self.head.parameters():
            if p in self.scores:
                score = self.scores[p]

                if torch.all(score == 0):
                    print(f"[Warning] All scores are zero for parameter of shape {p.shape}.")
                    masks.append(torch.ones_like(p, device=self.device))
                    continue

                # Compute the pruning threshold
                threshold = torch.quantile(score.flatten(), sparsity)
                mask = (score >= threshold).float()

                # Ensure a minimum of 5% of weights are kept
                keep_percent = mask.sum() / mask.numel()
                if keep_percent < 0.05:
                    print(f"[Warning] Only {keep_percent:.2%} of weights are kept, adjusting...")
                    top_k = max(int(0.05 * mask.numel()), 1)
                    values, _ = torch.topk(score.flatten(), top_k)
                    threshold = values.min()
                    mask = (score >= threshold).float()

                masks.append(mask)

        return masks



def iterative_pruning(pruner, dataloader, rounds=4, final_sparsity=0.9, num_batches=3):
    """
    Perform iterative pruning over multiple rounds, progressively increasing sparsity.

    Args:
        pruner (TaLoSPruner): Instance of the TaLoSPruner.
        dataloader (DataLoader): DataLoader for scoring.
        rounds (int): Number of pruning rounds.
        final_sparsity (float): Final sparsity ratio (0 < sparsity < 1).
        num_batches (int): Number of batches to score for each round.

    Returns:
        list: Final binary masks after iterative pruning.
    """
    keep_ratio = 1.0 - final_sparsity

    for r in range(rounds):
        current_keep = keep_ratio ** ((r + 1) / rounds)
        current_sparsity = 1.0 - current_keep
        print(f"[Round {r + 1}/{rounds}] Target sparsity: {current_sparsity:.4f}")

        # Compute Fisher scores
        pruner.score(dataloader, num_batches=num_batches)

        # Generate masks based on the scores
        masks = pruner.generate_masks(sparsity=current_sparsity)

    return masks

