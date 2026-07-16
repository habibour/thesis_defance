"""
Custom HF Trainer that supports:
  - class-weighted cross-entropy loss (for the imbalanced 3-class task)
  - optional FGM adversarial training step

Both are independently toggleable so each can be its own ablation row.
Works the same whether the underlying model is a full fine-tune or a
peft-wrapped LoRA model -- FGM's name-matching on "word_embeddings" still
finds the base model's embedding table under a LoRA wrapper since LoRA
doesn't touch the embedding layer by default.

Note: this implementation keeps things simple for a thesis-scale ablation
study -- it does not attempt to correctly rescale loss under gradient
accumulation > 1. Run with gradient_accumulation_steps=1 if use_fgm=True.
"""

import torch.nn as nn
from transformers import Trainer

from fgm import FGM


class WeightedFGMTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights=None,
        use_fgm: bool = False,
        fgm_epsilon: float = 1.0,
        fgm_emb_name: str = "word_embeddings",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_fgm = use_fgm
        self._fgm = FGM(self.model, emb_name=fgm_emb_name, epsilon=fgm_epsilon) if use_fgm else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device).to(logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _backward(self, loss):
        # transformers >= 4.30ish routes backward through accelerate; older
        # versions don't have self.accelerator. Support both.
        if hasattr(self, "accelerator"):
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        self._backward(loss)

        if self.use_fgm:
            self._fgm.attack()
            loss_adv = self.compute_loss(model, inputs)
            self._backward(loss_adv)
            self._fgm.restore()

        return loss.detach()
