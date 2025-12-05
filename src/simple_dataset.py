"""simple dataset without pytorch lightning."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class SimpleEventDataset(Dataset):
    """event sequence dataset with fixed-length padding."""

    def __init__(
        self,
        sequences: List[Dict],
        max_seq_length: int,
        event_to_idx: Dict[str, int],
        value_mean: float,
        value_std: float
    ):
        self.sequences = sequences
        self.max_seq_length = max_seq_length
        self.event_to_idx = event_to_idx
        self.value_mean = value_mean
        self.value_std = value_std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        event_types = seq["event_types"]
        event_values = seq["event_values"]
        time_deltas = seq["time_deltas"]
        label = seq["label"]

        # truncate if too long
        if len(event_types) > self.max_seq_length:
            event_types = event_types[-self.max_seq_length:]
            event_values = event_values[-self.max_seq_length:]
            time_deltas = time_deltas[-self.max_seq_length:]

        seq_len = len(event_types)

        # pad if too short
        if seq_len < self.max_seq_length:
            pad_len = self.max_seq_length - seq_len
            event_types = event_types + [0] * pad_len
            event_values = event_values + [0.0] * pad_len
            time_deltas = time_deltas + [1.0] * pad_len

        padding_mask = [False] * seq_len + [True] * (self.max_seq_length - seq_len)

        return {
            "event_types": torch.tensor(event_types, dtype=torch.long),
            "event_values": torch.tensor(event_values, dtype=torch.float32),
            "time_deltas": torch.tensor(time_deltas, dtype=torch.float32),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_and_split_data(events_path: str, max_seq_length: int = 64):
    """load events and create train/val/test datasets."""

    # load data
    df = pd.read_csv(events_path)
    df["timestamp"] = pd.to_datetime(df["event_timestamp"])
    df = df.sort_values(["customer_id", "timestamp"])

    # create event mapping
    event_types = sorted(df["event_type"].unique())
    event_to_idx = {event: idx for idx, event in enumerate(event_types)}

    # compute stats
    value_mean = df["event_value"].mean()
    value_std = df["event_value"].std()

    # create sequences
    sequences = []
    for customer_id, group in df.groupby("customer_id"):
        group = group.sort_values("timestamp")
        label = group["label_churn_30d"].iloc[0]

        event_types_seq = [event_to_idx[e] for e in group["event_type"]]
        event_values_seq = ((group["event_value"] - value_mean) / value_std).tolist()

        timestamps = group["timestamp"].values
        time_deltas_seq = []
        for i in range(len(timestamps)):
            if i == 0:
                delta = 3600.0
            else:
                delta = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
                delta = max(delta, 1.0)
            time_deltas_seq.append(delta)

        sequences.append({
            "customer_id": customer_id,
            "event_types": event_types_seq,
            "event_values": event_values_seq,
            "time_deltas": time_deltas_seq,
            "label": label
        })

    # split
    rng = np.random.RandomState(42)
    indices = np.arange(len(sequences))
    rng.shuffle(indices)

    n_train = int(len(sequences) * 0.7)
    n_val = int(len(sequences) * 0.15)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    test_seqs = [sequences[i] for i in test_idx]

    # create datasets
    train_dataset = SimpleEventDataset(train_seqs, max_seq_length, event_to_idx, value_mean, value_std)
    val_dataset = SimpleEventDataset(val_seqs, max_seq_length, event_to_idx, value_mean, value_std)
    test_dataset = SimpleEventDataset(test_seqs, max_seq_length, event_to_idx, value_mean, value_std)

    return train_dataset, val_dataset, test_dataset, event_to_idx, value_mean, value_std
