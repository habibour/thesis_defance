"""
Fast Gradient Method (FGM) adversarial training.

Perturbs the word embedding layer along the gradient direction to force the
model to learn representations that are robust to small input perturbations.
This is the adversarial training technique used by the Knowdee BLP-2023
Task 2 system on top of BanglaBERT-large; here it's one of the ablation
steps on the Bengali_Sentiment benchmark.

Reference technique (standard, not paper-specific): Miyato et al., 2017,
"Adversarial Training Methods for Semi-Supervised Text Classification".
"""

import torch


class FGM:
    def __init__(self, model: torch.nn.Module, emb_name: str = "word_embeddings", epsilon: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
