import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Mô hình LSTM để dự đoán giá đóng cửa trong pred_len ngày tiếp theo.
    - input_size: số feature đầu vào (ví dụ Close, MA7, RSI14,…)
    - hidden_size: số đơn vị ẩn của LSTM
    - num_layers: số layer LSTM chồng lên nhau
    - pred_len: số bước cần dự đoán (ở đây là 5)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 126,
                 num_layers: int = 2,
                 pred_len: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]     # lấy hidden state của bước cuối
        y_pred = self.fc(last_hidden)   # (batch, pred_len)
        return y_pred