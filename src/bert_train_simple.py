"""
simple training script for time-aware transformer encoder (no lightning).
uses standard pytorch training loop with early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, average_precision_score

from src.bert_config import EncoderConfig
from src.simple_dataset import load_and_split_data


# simplified model without lightning
class SimpleTimeAwareEncoder(nn.Module):
    """time-aware encoder without pytorch lightning dependencies."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        from src.bert_layers import (
            SinusoidalTimeEncoding,
            FocalLoss,
            PreNormTransformerBlock,
            MeanPooling
        )

        # input embeddings
        self.event_embedding = nn.Embedding(config.num_event_types, config.embedding_dim)
        self.time_encoder = SinusoidalTimeEncoding(config.time_encoding_dim)
        self.value_projection = nn.Linear(1, config.value_projection_dim)

        input_dim = config.embedding_dim + config.time_encoding_dim + config.value_projection_dim
        self.input_projection = nn.Linear(input_dim, config.embedding_dim)

        # transformer
        self.transformer_layers = nn.ModuleList([
            PreNormTransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.feedforward_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.pooling = MeanPooling()

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim // 2, 2)
        )

        self.criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, event_types, event_values, time_deltas, padding_mask):
        event_embeds = self.event_embedding(event_types)
        time_embeds = self.time_encoder(time_deltas)
        value_embeds = self.value_projection(event_values.unsqueeze(-1))

        combined = torch.cat([event_embeds, time_embeds, value_embeds], dim=-1)
        x = self.input_projection(combined)

        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        x = self.final_norm(x)
        valid_mask = ~padding_mask
        user_embedding = self.pooling(x, valid_mask)

        logits = self.classifier(user_embedding)
        return logits


def train_epoch(model, dataloader, optimizer, device):
    """train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="training"):
        event_types = batch["event_types"].to(device)
        event_values = batch["event_values"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(event_types, event_values, time_deltas, padding_mask)
        loss = model.criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # collect predictions
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return avg_loss, auroc, auprc


@torch.no_grad()
def evaluate(model, dataloader, device):
    """evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="evaluating"):
        event_types = batch["event_types"].to(device)
        event_values = batch["event_values"].to(device)
        time_deltas = batch["time_deltas"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(event_types, event_values, time_deltas, padding_mask)
        loss = model.criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return avg_loss, auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-path", type=str, default="events.csv")
    parser.add_argument("--output-dir", type=str, default="bert_models")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # setup
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # create config
    config = EncoderConfig(max_epochs=args.max_epochs, patience=args.patience)

    # setup data
    print("loading data...")
    train_dataset, val_dataset, test_dataset, event_to_idx, value_mean, value_std = load_and_split_data(
        args.events_path, config.max_seq_length
    )

    config.num_event_types = len(event_to_idx)

    print(f"event types: {config.num_event_types}")
    print(f"max sequence length: {config.max_seq_length}")
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # create model
    print("\ncreating model...")
    model = SimpleTimeAwareEncoder(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {num_params:,}")

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs
    )

    # training loop
    print(f"\ntraining for up to {args.max_epochs} epochs...")
    best_auprc = 0
    patience_counter = 0
    history = []

    for epoch in range(args.max_epochs):
        print(f"\nepoch {epoch+1}/{args.max_epochs}")

        # train
        train_loss, train_auroc, train_auprc = train_epoch(model, train_loader, optimizer, device)
        print(f"  train: loss={train_loss:.4f}, auroc={train_auroc:.4f}, auprc={train_auprc:.4f}")

        # validate
        val_loss, val_auroc, val_auprc = evaluate(model, val_loader, device)
        print(f"  val:   loss={val_loss:.4f}, auroc={val_auroc:.4f}, auprc={val_auprc:.4f}")

        scheduler.step()

        # save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_auroc": train_auroc,
            "train_auprc": train_auprc,
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc
        })

        # check for improvement
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            patience_counter = 0

            # save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": val_auprc,
                "config": config
            }, output_path / "best_model.pt")
            print(f"  saved best model (auprc: {val_auprc:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print("\nearly stopping triggered")
                break

    # test on best model
    print("\n\ntesting on best model...")
    checkpoint = torch.load(output_path / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_auroc, test_auprc = evaluate(model, test_loader, device)
    print(f"\ntest results:")
    print(f"  loss: {test_loss:.4f}")
    print(f"  auroc: {test_auroc:.4f}")
    print(f"  auprc: {test_auprc:.4f}")

    # save history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\ntraining complete. saved to {output_path}")


if __name__ == "__main__":
    main()
