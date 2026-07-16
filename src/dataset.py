"""
Dataset loading for the Bengali_Sentiment benchmark (Islam et al., 2020).

Expects two CSV files with columns: Data, Sentiment
where Sentiment in {0, 1, 2} = {Neutral, Positive, Negative}
(mapping confirmed by matching per-class counts against the base paper's
Table III: Negative=7071, Positive=3926, Neutral=3855 in the combined
train+valid split).

For the 2-class task we drop Neutral (label 0) and remap the remainder to
{0: Positive, 1: Negative}, mirroring how the base paper constructs its
13,120-row 2-class dataset from the 17,852-row 3-class one.
"""

from dataclasses import dataclass, field

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from preprocessing import normalize_text

LABEL_NAMES_3CLASS = ["Neutral", "Positive", "Negative"]
LABEL_NAMES_2CLASS = ["Positive", "Negative"]


@dataclass
class DatasetBundle:
    dataset_dict: DatasetDict
    num_labels: int
    label_names: list = field(default_factory=list)


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Data", "Sentiment"]).copy()
    df["Sentiment"] = df["Sentiment"].astype(int)
    df["Data"] = df["Data"].astype(str)
    return df


def build_datasets(
    train_path: str,
    test_path: str,
    task: str = "3class",
    val_size: float = 0.1,
    seed: int = 42,
    apply_normalization: bool = True,
) -> DatasetBundle:
    """Build a DatasetDict with train/validation/test splits.

    `task` is one of "3class" or "2class".
    Validation is carved out of the training file only (stratified, 10% by
    default) since only a combined train file and a held-out test file are
    provided; the test file is never touched until final evaluation.
    """
    assert task in ("3class", "2class")

    train_df = _load_raw(train_path)
    test_df = _load_raw(test_path)

    if task == "2class":
        train_df = train_df[train_df["Sentiment"] != 0].copy()
        test_df = test_df[test_df["Sentiment"] != 0].copy()
        remap = {1: 0, 2: 1}  # Positive -> 0, Negative -> 1
        train_df["label"] = train_df["Sentiment"].map(remap)
        test_df["label"] = test_df["Sentiment"].map(remap)
        num_labels = 2
        label_names = LABEL_NAMES_2CLASS
    else:
        train_df["label"] = train_df["Sentiment"]
        test_df["label"] = test_df["Sentiment"]
        num_labels = 3
        label_names = LABEL_NAMES_3CLASS

    if apply_normalization:
        train_df["text"] = train_df["Data"].map(normalize_text)
        test_df["text"] = test_df["Data"].map(normalize_text)
    else:
        train_df["text"] = train_df["Data"]
        test_df["text"] = test_df["Data"]

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"],
    )

    cols = ["text", "label"]
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[cols].reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df[cols].reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df[cols].reset_index(drop=True)),
        }
    )

    return DatasetBundle(dataset_dict=dataset_dict, num_labels=num_labels, label_names=label_names)


def class_weights_from_labels(labels, num_labels: int):
    """Inverse-frequency class weights for CrossEntropyLoss, normalized so
    they average to 1 (keeps loss magnitude comparable to unweighted CE).
    """
    import numpy as np
    import torch

    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0  # avoid div-by-zero for an absent class
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return torch.tensor(weights, dtype=torch.float)
