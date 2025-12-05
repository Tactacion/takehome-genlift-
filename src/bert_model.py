"""
time-aware transformer encoder for user representation learning.
learns dense embeddings h_u ∈ R^d that capture user behavior patterns.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple
from torchmetrics import AUROC, AveragePrecision, Accuracy, F1Score

from src.bert_config import EncoderConfig
from src.bert_layers import (
    SinusoidalTimeEncoding,
    FocalLoss,
    PreNormTransformerBlock,
    MeanPooling,
    MaxPooling
)


class TimeAwareEncoder(pl.LightningModule):
    """
    time-aware transformer encoder for churn prediction.

    architecture philosophy:
    instead of generating future events (generative modeling), we learn
    a dense representation of each customer by encoding their event history.
    this representation h_u captures their position in a learned "behavior manifold".

    churning users cluster in one region of this manifold, active users in another.
    a simple linear classifier can then separate these regions.

    this approach is more robust on small datasets because:
    1. discriminative (not generative) - easier optimization
    2. focal loss handles class imbalance
    3. pre-norm transformers stabilize training
    4. mean pooling provides noise robustness
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # input embeddings
        self.event_embedding = nn.Embedding(
            config.num_event_types,
            config.embedding_dim
        )

        # time encoding: log(time) -> sinusoidal features
        self.time_encoder = SinusoidalTimeEncoding(config.time_encoding_dim)

        # value projection: continuous value -> dense features
        self.value_projection = nn.Linear(1, config.value_projection_dim)

        # combine all input features
        input_dim = (
            config.embedding_dim +  # event type
            config.time_encoding_dim +  # time delta
            config.value_projection_dim  # event value
        )

        # project to transformer dimension
        self.input_projection = nn.Linear(input_dim, config.embedding_dim)

        # transformer encoder
        self.transformer_layers = nn.ModuleList([
            PreNormTransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # final layer norm (standard in pre-norm architectures)
        self.final_norm = nn.LayerNorm(config.embedding_dim)

        # pooling strategy
        if config.pooling == "mean":
            self.pooling = MeanPooling()
        elif config.pooling == "max":
            self.pooling = MaxPooling()
        else:
            raise ValueError(f"unknown pooling: {config.pooling}")

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, 2)  # binary classification
        )

        # loss function
        self.criterion = FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha
        )

        # metrics
        self.train_auroc = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        event_types: torch.Tensor,
        event_values: torch.Tensor,
        time_deltas: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        encode event sequence into dense representation.

        this is the core representation learning: we map a variable-length
        sequence of events into a fixed-size vector h_u that captures the
        user's behavioral signature.

        args:
            event_types: (batch, seq_len) discrete event type indices
            event_values: (batch, seq_len) continuous event values
            time_deltas: (batch, seq_len) time since previous event
            padding_mask: (batch, seq_len) boolean (true = padding)

        returns:
            (batch, embedding_dim) user representation
        """
        # embed event types
        event_embeds = self.event_embedding(event_types)

        # encode time deltas (log + sinusoidal)
        time_embeds = self.time_encoder(time_deltas)

        # project event values
        value_embeds = self.value_projection(event_values.unsqueeze(-1))

        # concatenate all features
        combined = torch.cat([event_embeds, time_embeds, value_embeds], dim=-1)

        # project to transformer dimension
        x = self.input_projection(combined)

        # pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        # final normalization
        x = self.final_norm(x)

        # pool across sequence dimension
        # this aggregates the entire event history into a single vector
        # geometric intuition: finding the user's "center of mass" in embedding space
        valid_mask = ~padding_mask  # invert for pooling (true = valid)
        user_embedding = self.pooling(x, valid_mask)

        return user_embedding

    def forward(
        self,
        event_types: torch.Tensor,
        event_values: torch.Tensor,
        time_deltas: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        full forward pass: encode sequence and classify.

        returns:
            (batch, 2) logits for [no_churn, churn]
        """
        # encode to dense representation
        user_embedding = self.encode(
            event_types, event_values, time_deltas, padding_mask
        )

        # classify
        logits = self.classifier(user_embedding)

        return logits

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """training step with focal loss."""
        event_types = batch["event_types"]
        event_values = batch["event_values"]
        time_deltas = batch["time_deltas"]
        padding_mask = batch["padding_mask"]
        labels = batch["labels"]

        # forward pass
        logits = self(event_types, event_values, time_deltas, padding_mask)

        # compute loss
        loss = self.criterion(logits, labels)

        # compute metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]  # probability of churn
        self.train_auroc(probs, labels)
        self.train_auprc(probs, labels)

        # log
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auroc", self.train_auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_auprc", self.train_auprc, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """validation step."""
        event_types = batch["event_types"]
        event_values = batch["event_values"]
        time_deltas = batch["time_deltas"]
        padding_mask = batch["padding_mask"]
        labels = batch["labels"]

        # forward pass
        logits = self(event_types, event_values, time_deltas, padding_mask)

        # compute loss
        loss = self.criterion(logits, labels)

        # compute metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.val_auroc(probs, labels)
        self.val_auprc(probs, labels)

        # log
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auprc", self.val_auprc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """test step."""
        event_types = batch["event_types"]
        event_values = batch["event_values"]
        time_deltas = batch["time_deltas"]
        padding_mask = batch["padding_mask"]
        labels = batch["labels"]

        # forward pass
        logits = self(event_types, event_values, time_deltas, padding_mask)

        # compute loss
        loss = self.criterion(logits, labels)

        # compute metrics
        probs = torch.softmax(logits, dim=-1)[:, 1]
        self.test_auroc(probs, labels)
        self.test_auprc(probs, labels)

        # log
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True)
        self.log("test_auprc", self.test_auprc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def predict_step(self, batch: Dict, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """prediction step - returns probabilities and embeddings."""
        event_types = batch["event_types"]
        event_values = batch["event_values"]
        time_deltas = batch["time_deltas"]
        padding_mask = batch["padding_mask"]

        # get embeddings
        embeddings = self.encode(
            event_types, event_values, time_deltas, padding_mask
        )

        # get predictions
        logits = self.classifier(embeddings)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # churn probability

        return probs, embeddings
