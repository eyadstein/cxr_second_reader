"""Shared helpers for the experimental notebooks (not clinical software)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.data import load_table, resolve_image


def study_rows(split: str, max_studies: int | None = None) -> pd.DataFrame:
    """Return one frontal image and nonblank report per study for a split."""
    frame = load_table(split)
    if "projection" in frame.columns:
        frontal = frame["projection"].fillna("").astype(str).str.lower().isin(
            {"frontal", "ap", "pa"}
        )
        frame = frame.loc[frontal].copy()
    frame["target_text"] = frame["target_text"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["target_text"].ne("")]
    frame = frame.drop_duplicates("uid", keep="first").reset_index(drop=True)
    if max_studies is not None:
        frame = frame.head(max_studies).copy()
    if frame.empty:
        raise ValueError(f"No usable frontal studies were found for split={split!r}.")
    return frame


class ImageTextRows(Dataset):
    """Lazy image/report rows using the shared split and image resolver."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = resolve_image(row)
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "text": str(row["target_text"]),
            "uid": str(row["uid"]),
            "filename": str(row.get("filename", Path(path).name)),
            "indication": str(row.get("indication", "") or ""),
            "class": str(row.get("class", "")),
        }
