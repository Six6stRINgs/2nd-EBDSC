import torch.nn as nn
from einops import rearrange


class Configs:
    def __init__(self):
        self.d_model = 640  # 对应 5 * 128 (M*D)
        self.e_layers = 3
        self.dropout = 0.3
        self.num_classes = 12
        self.dim = 256      # 内部映射维度，防止从 640 压缩到 5 丢失太多细节


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

        # 1. 初始特征映射：从 640 (拼接后的 Wide-Value) 映射到更合理的特征空间
        self.input_proj = nn.Sequential(
            nn.Linear(self.d_model, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # 2. 核心双向 LSTM
        # 注意：为了支持残差连接，我们将 hidden_dim * 2 设为与 dim 一致，或进行额外的投影
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim // 2, # 这样 BiLSTM 输出正好是 dim
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True,
        )

        # 3. 残差层：如果维度匹配则相加
        self.res_proj = nn.Identity()

        # 优化后的 MLP Header
        self.ln = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, self.num_classes),
        )

    def forward(self, x):
        if self.wide_value_emb or len(x.shape) == 4:
            # [B, L, 5, 128] -> [B, L, 640]
            x = rearrange(x, "b l m d -> b l (m d)")

        # 1. 特征投影
        x_proj = self.input_proj(x)

        # 2. LSTM
        out, _ = self.lstm(x_proj)
        
        # 3. 残差连接 (Projected Input + LSTM Output)
        out = out + self.res_proj(x_proj)

        # 4. 优化后的 MLP Header
        out = self.ln(out)
        out = self.mlp(out)
        return out
