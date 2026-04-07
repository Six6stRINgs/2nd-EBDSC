import torch.nn as nn
from einops import rearrange


class Configs:
    def __init__(self):
        self.d_model = 640
        self.e_layers = 2
        self.dropout = 0.5
        self.num_classes = 12
        self.dim = 5


class BiLSTM(nn.Module):
    def __init__(self, configs, wide_value_emb=False):
        super(BiLSTM, self).__init__()
        self.d_model = configs.d_model
        self.num_classes = (
            configs.num_classes if hasattr(configs, "num_classes") else 12
        )
        self.hidden_dim = configs.d_model // 2
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.wide_value_emb = wide_value_emb
        self.dim = configs.dim

        self.input_linear = nn.Linear(self.d_model, self.dim)

        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True,
        )

        # 优化后的 MLP Header
        self.ln = nn.LayerNorm(self.hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.num_classes),
        )

    def forward(self, x):
        if self.wide_value_emb or len(x.shape) == 4:
            x = rearrange(x, "b l m d -> b l (m d)")

        x = self.input_linear(x)

        # LSTM 输出 shape: [B, L, hidden_dim * 2]
        out, _ = self.lstm(x)

        # 应用归一化和 MLP
        out = self.ln(out)
        out = self.mlp(out)
        return out
