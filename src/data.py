import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.config import SHARED

def load_table(split=None):
    df = pd.read_csv(f"{SHARED}/dataset.csv").fillna("")
    return df if split is None else df[df.split == split].reset_index(drop=True)

class CXRDataset(Dataset):
    def __init__(self, split, size=224, transform=None):
        self.df, self.size, self.transform = load_table(split), size, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{SHARED}/img{self.size}/{r.uid}.png").convert("L")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "indication": r.indication, "target_text": r.target_text, "uid": r.uid}