import numpy as np
import torch
import torch.utils.data as Data


class ExtraDataset(Data.Dataset):
    """自定义数据集函数"""

    def __init__(
        self,
        base_dataset,
        hard=None,
        pos_d=None,
        if_emb=True,
        f_mask=lambda x: torch.rand_like(x) * 2 - 1,
        config=None,
    ):
        super(ExtraDataset, self).__init__()
        self.base_dataset = base_dataset
        self.use_embedding = if_emb
        self.hard = hard
        self.f_mask = f_mask

        self.norm_eps = 1e-8
        self.toa_rescale_divisor = 5e5
        self.toa_rescale_bias = 0.0002
        self.toa_rescale_scale = 0.0005
        self.rf_prob = 1.0
        self.pw_prob = 1.0
        self.pa_prob = 0.1
        self.doa_prob = 0.5
        self.rf_mask_range = (5, 60)
        self.pw_mask_range = (6, 100)
        self.doa_mask_range = (6, 14)

        if config is not None:
            wide_cfg = config.wide_value_embedding
            mask_cfg = config.masking
            self.norm_eps = wide_cfg.norm_eps
            self.toa_rescale_divisor = wide_cfg.toa_rescale_divisor
            self.toa_rescale_bias = wide_cfg.toa_rescale_bias
            self.toa_rescale_scale = wide_cfg.toa_rescale_scale
            pos_d = wide_cfg.pos_d if pos_d is None else pos_d
            self.d_step = wide_cfg.d_step
            mod_max = wide_cfg.mod_max
            self.rf_prob = mask_cfg.rf_prob
            self.pw_prob = mask_cfg.pw_prob
            self.pa_prob = mask_cfg.pa_prob
            self.doa_prob = mask_cfg.doa_prob
            self.rf_mask_range = tuple(mask_cfg.rf_mask_range)
            self.pw_mask_range = tuple(mask_cfg.pw_mask_range)
            self.doa_mask_range = tuple(mask_cfg.doa_mask_range)
        else:
            pos_d = 128 if pos_d is None else pos_d
            self.d_step = 8
            mod_max = 65536

        if not self.use_embedding:
            return

        self.d_model = pos_d
        self.div_term = 1.0 / (
            mod_max ** (torch.arange(0, self.d_model, self.d_step) / self.d_model)
        )
        self.d_mod = lambda m: np.floor(self.d_step * np.log2(m)).astype(np.int64) + 1

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        inputs, target = self.base_dataset[idx]

        if not self.use_embedding:
            inputs = (inputs - inputs.mean(axis=0)) / (inputs.std(axis=0) + self.norm_eps)
            inputs[:, 0] = (
                (inputs[:, 0] / self.toa_rescale_divisor) - self.toa_rescale_bias
            ) / self.toa_rescale_scale
            return inputs, target

        positions: torch.FloatTensor = inputs
        win_size, input_channels = positions.size()
        pe = torch.zeros(win_size, input_channels, self.d_model)

        positions = positions.unsqueeze(-1)

        for i in range(self.d_step):
            pe[:, :, i :: self.d_step] = (
                positions * self.div_term + 1 / self.d_step * i
            ) % 1 * 2 - 1

        if self.hard:
            if np.random.rand() < self.rf_prob * self.hard:
                mask_d_min = np.random.randint(
                    self.d_mod(self.rf_mask_range[0]), self.d_mod(self.rf_mask_range[1])
                )
                pe[:, 1, mask_d_min:] = self.f_mask(pe[:, 1, mask_d_min:])

            if np.random.rand() < self.pw_prob * self.hard:
                mask_d_min = np.random.randint(
                    self.d_mod(self.pw_mask_range[0]), self.d_mod(self.pw_mask_range[1])
                )
                pe[:, 2, mask_d_min:] = self.f_mask(pe[:, 2, mask_d_min:])

            if np.random.rand() < self.pa_prob * self.hard:
                pe[:, 3, :] = self.f_mask(pe[:, 3, :])

            if np.random.rand() < self.doa_prob * self.hard:
                mask_d_min = np.random.randint(
                    self.d_mod(self.doa_mask_range[0]), self.d_mod(self.doa_mask_range[1])
                )
                pe[:, 4, mask_d_min:] = self.f_mask(pe[:, 4, mask_d_min:])

        return pe, target


class ExtraDataset_woEmb(ExtraDataset):
    """不使用宽值域嵌入的数据集包装"""

    def __init__(self, base_dataset, hard=None, pos_d=None, f_mask=None, config=None):
        super(ExtraDataset_woEmb, self).__init__(
            base_dataset,
            hard=hard,
            pos_d=pos_d,
            if_emb=False,
            f_mask=f_mask,
            config=config,
        )
