import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import device


class CustomDataset(Dataset):
    def __init__(self, data_path: str, target: str = 'Survived'):
        self.df = pd.read_csv(data_path)
        self.target = self.df[target]
        self.X = self.df.drop(columns=[target], axis=1)

    def __getitem__(self, index: int):
        x = self.X.iloc[index]
        y = self.target.iloc[index]

        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.df)


def dataloaders(train_path: str, test_path: str, batch_size: int):
    train_data = CustomDataset(train_path)
    test_data = CustomDataset(test_path)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader
