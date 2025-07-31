import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device=torch.device('cpu')):
        # X: numpy array (n_samples, seq_len, n_features)
        # y: numpy array (n_samples, pred_len)
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(X: np.ndarray,
                    y: np.ndarray,
                    batch_size: int = 32,
                    split_ratio: float = 0.8,
                    shuffle_train: bool = True,
                    device: torch.device = torch.device('cpu')):
    """
    Trả về train_loader và test_loader.
    - split_ratio: tỉ lệ phần trăm dữ liệu train (ví dụ 0.8)
    - batch_size: kích thước batch
    """
    # Chia train/test
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Tạo Dataset
    train_ds = StockDataset(X_train, y_train, device=device)
    test_ds = StockDataset(X_test, y_test, device=device)

    # Tạo DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader