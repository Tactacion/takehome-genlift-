"""
components for time-aware transformer encoder.
includes sinusoidal time encoding, focal loss, and pre-norm transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEncoding(nn.Module):
    """
    sinusoidal positional encoding applied to log(time_delta).

    this provides scale-invariant time awareness - the model can distinguish
    between a 5-second gap and a 5-day gap via the log transformation,
    then encodes it into a high-dimensional space using sin/cos functions.

    based on "attention is all you need" (vaswani et al.) but applied to
    continuous time rather than discrete positions.
    """

    def __init__(self, encoding_dim: int = 32):
        super().__init__()
        self.encoding_dim = encoding_dim

        # create frequency bands for sin/cos encoding
        # frequencies decrease exponentially
        inv_freq = 1.0 / (10000 ** (torch.arange(0, encoding_dim, 2).float() / encoding_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        encode time deltas into sinusoidal features.

        args:
            time_deltas: (batch, seq_len) time gaps in seconds

        returns:
            (batch, seq_len, encoding_dim) encoded time features
        """
        # take log for scale invariance (avoid log(0))
        log_time = torch.log(time_deltas + 1.0)

        # compute angles: (batch, seq, 1) * (encoding_dim/2,) -> (batch, seq, encoding_dim/2)
        angles = log_time.unsqueeze(-1) * self.inv_freq

        # apply sin/cos and concatenate
        sin_encoding = torch.sin(angles)
        cos_encoding = torch.cos(angles)

        return torch.cat([sin_encoding, cos_encoding], dim=-1)


class FocalLoss(nn.Module):
    """
    focal loss for addressing class imbalance.

    standard cross entropy treats all examples equally. when churn is rare
    (10-30% of users), the model can achieve high accuracy by predicting
    "no churn" for everyone. focal loss down-weights easy examples and
    focuses on hard, misclassified cases.

    formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t is the probability of the correct class
    - γ (gamma) is the focusing parameter (higher = more focus on hard cases)
    - α_t is the class weight

    reference: "focal loss for dense object detection" (lin et al., 2017)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None):
        """
        args:
            gamma: focusing parameter. higher values increase focus on hard examples.
                   typical values: 2.0 (strong focus), 1.0 (moderate), 0.0 (no focus = CE)
            alpha: class weight for positive class. if none, uses equal weighting.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        compute focal loss.

        args:
            logits: (batch, num_classes) raw model outputs
            targets: (batch,) ground truth class indices

        returns:
            scalar loss value
        """
        # compute probabilities
        probs = F.softmax(logits, dim=-1)

        # gather probability of correct class
        batch_size = targets.size(0)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_onehot).sum(dim=-1)

        # compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # compute cross entropy term
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # apply class weights if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets_onehot[:, 1] + (1 - self.alpha) * targets_onehot[:, 0]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class PreNormTransformerBlock(nn.Module):
    """
    transformer encoder block with pre-normalization.

    in standard transformers, layer norm comes after attention/ffn (post-norm).
    pre-norm architecture applies layer norm before these operations, which:
    1. stabilizes gradients (easier optimization)
    2. allows deeper networks without careful init
    3. works better on smaller datasets

    architecture:
        x = x + attention(layernorm(x))
        x = x + ffn(layernorm(x))

    reference: "on layer normalization in the transformer architecture" (xiong et al., 2020)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        args:
            x: (batch, seq_len, embedding_dim)
            padding_mask: (batch, seq_len) boolean mask (true = padding)

        returns:
            (batch, seq_len, embedding_dim)
        """
        # pre-norm attention
        normed = self.ln1(x)
        attn_output, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        x = x + attn_output

        # pre-norm feedforward
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + ffn_output

        return x


class MeanPooling(nn.Module):
    """
    mean pooling across sequence dimension.

    aggregates variable-length sequences into a single fixed-size vector
    by taking the average. this is more robust than using a cls token
    or max pooling, especially for noisy event data.

    the geometric intuition: we're finding the "center of mass" of the
    user's trajectory in the learned embedding space. this smooths out
    noise from individual events.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        args:
            x: (batch, seq_len, embedding_dim) sequence embeddings
            mask: (batch, seq_len) boolean mask (true = valid, false = padding)

        returns:
            (batch, embedding_dim) pooled representation
        """
        if mask is None:
            # simple mean if no mask
            return x.mean(dim=1)

        # mask out padding tokens
        mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_x = x * mask

        # sum and divide by number of valid tokens
        sum_embeddings = masked_x.sum(dim=1)
        num_valid = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero

        return sum_embeddings / num_valid


class MaxPooling(nn.Module):
    """
    max pooling across sequence dimension.

    takes the maximum value for each feature dimension. this captures
    the most "extreme" or "activated" features from the sequence.

    geometric intuition: finds the furthest point reached by the user
    in each dimension of the embedding manifold.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        args:
            x: (batch, seq_len, embedding_dim)
            mask: (batch, seq_len) boolean mask (true = valid)

        returns:
            (batch, embedding_dim)
        """
        if mask is not None:
            # set padding positions to very negative value
            mask = mask.unsqueeze(-1).float()
            x = x.masked_fill(mask == 0, -1e9)

        return x.max(dim=1)[0]
