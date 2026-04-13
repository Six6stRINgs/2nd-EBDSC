import platform
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from joblib import Parallel, delayed

TAG_LEN = 12
WINDOW_SIZE = 1024

if platform.system() == "Windows":
    DATASET_DIR = "F:/datasets/extra"
else:
    DATASET_DIR = "/root/autodl-tmp/datasets/extra"
DATA_NAME = "PosAll"


def configure_data_globals(
    tag_len: int = None,
    window_size: int = None,
    dataset_dir: str = None,
    data_name: str = None,
):
    """允许运行时从外部覆盖数据模块的全局配置。"""
    global TAG_LEN, WINDOW_SIZE, DATASET_DIR, DATA_NAME

    if tag_len is not None:
        TAG_LEN = tag_len
    if window_size is not None:
        WINDOW_SIZE = window_size
    if dataset_dir is not None:
        DATASET_DIR = dataset_dir
    if data_name is not None:
        DATA_NAME = data_name


def get_data_globals():
    """返回当前数据模块使用中的全局配置。"""
    return {
        "TAG_LEN": TAG_LEN,
        "WINDOW_SIZE": WINDOW_SIZE,
        "DATASET_DIR": DATASET_DIR,
        "DATA_NAME": DATA_NAME,
    }


def read_txt(filepath):
    """读取 txt 文件，并以 DataFrame 返回"""
    columns = ["TOA", "RF", "PW", "PA", "DOA", "TAG"]
    df = pd.read_csv(filepath, sep=r"\s+", names=columns)
    df["TOA"] = df["TOA"].astype(np.float32)
    df["RF"] = df["RF"].astype(np.float32)
    df["PW"] = df["PW"].astype(np.float32)
    df["PA"] = df["PA"].astype(np.float32)
    df["DOA"] = df["DOA"].astype(np.float32)
    df["TAG"] = df["TAG"].astype(np.int32)
    return df


def read_train_data(file_index):
    return read_txt(f"{DATASET_DIR}/训练数据集/信号类型{file_index+1}训练集.txt")


def read_test_data(file_index):
    return read_txt(f"{DATASET_DIR}/验证数据集/验证集{file_index+1}.txt")


def read_dfs() -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """并行读取数据集"""
    print("并行读取数据集中...", end=" ")
    df_list: List[pd.DataFrame] = Parallel(n_jobs=-1)(
        delayed(read_train_data)(i) for i in range(TAG_LEN)
    )
    test_df_list: List[pd.DataFrame] = Parallel(n_jobs=-1)(
        delayed(read_test_data)(i) for i in range(3)
    )
    print("读取数据集 end")
    return df_list, test_df_list


def pre_reshape(
    df: pd.DataFrame,
    sample_ratio: float = None,
    drop_sig_ratio: float = 0.99,
    cut_len: float = 5.0,
    time_scale_mean: float = 1.0,
    time_scale_std: float = 0.2,
    clip_range: Tuple[float, float] = (0.5, 2.0),
    time_shift_exp_scale: float = 1.0,
) -> pd.DataFrame:
    """数据时间尺度上变化，模拟频移等增强"""
    start = np.random.uniform(df["TOA"].min(), df["TOA"].max())
    df_scaled: pd.DataFrame = df[
        (df["TOA"] >= start) & (df["TOA"] <= start + cut_len)
    ].copy()

    if sample_ratio is None:
        df_scaled = df_scaled.sample(frac=np.random.uniform(drop_sig_ratio, 1))
    else:
        df_scaled = df_scaled.sample(frac=sample_ratio)

    df_scaled["TOA"] *= np.clip(
        np.random.normal(time_scale_mean, time_scale_std), *clip_range
    )
    df_scaled["TOA"] -= df_scaled["TOA"].min()
    df_scaled["TOA"] += np.clip(np.random.exponential(time_shift_exp_scale), 0, cut_len)

    return df_scaled


def to_dict(
    df: pd.DataFrame,
    reshap=False,
    mode="default",
    noise=1.0,
    noise_snr=None,
    if_pri=False,
    rf_mod: int = 100,
    pw_mod: int = 10,
    doa_mod: int = 90,
    rf_scale_range: Tuple[float, float] = (0.2, 2.0),
    pw_scale_range: Tuple[float, float] = (0.5, 2.0),
    rf_offset_std: float = 200.0,
    pw_offset_std: float = 10.0,
    global_rf_shift_range: Tuple[float, float] = (-1000.0, 1000.0),
    global_pw_shift_range: Tuple[float, float] = (-10.0, 10.0),
    global_doa_shift_range: Tuple[float, float] = (-60.0, 60.0),
    toa_scale: float = 5e5,
    pri_bias: float = 0.0002,
    pri_scale: float = 0.0005,
    pw_multiplier: float = 10.0,
):
    """
    数据转换为模型输入格式
    mode: 'default' (线性缩放), 'manual' (模归一化)
    """

    def normalize(d: pd.Series) -> pd.Series:
        return (d - d.mean()) / (d.std() + 1e-8)

    def mod_normalize(d: pd.Series, mod: int) -> pd.Series:
        return (d % mod) / mod

    def add_noise(signal: pd.Series, snr: float):
        signal_power = signal.std() ** 2
        noise_power = signal_power / (10 ** (snr / 10))
        noise_vec = np.random.normal(0, np.sqrt(noise_power + 1e-10), len(signal))
        return (signal + noise_vec).astype(np.float32)

    noise_snr = np.inf if noise_snr is None else noise_snr

    df = df.copy()
    d_2 = pd.DataFrame()
    d_tag = pd.DataFrame()

    if reshap:
        for i in range(TAG_LEN):
            mask = df["TAG"] == i + 1
            if not mask.any():
                continue

            di = df.loc[mask, "RF"]
            scale = np.random.uniform(*rf_scale_range) * noise
            offset = (
                np.random.uniform(0, rf_mod) if mode == "manual" else np.random.normal(0, rf_offset_std)
            ) * noise
            df.loc[mask, "RF"] = scale * (di - di.mean()) + di.mean() + offset

            di = df.loc[mask, "PW"]
            scale = np.random.uniform(*pw_scale_range) * noise
            offset = (
                np.random.uniform(0, pw_mod) if mode == "manual" else np.random.normal(0, pw_offset_std)
            ) * noise
            df.loc[mask, "PW"] = scale * (di - di.mean()) + di.mean() + offset

            df.loc[mask, "DOA"] += np.random.uniform(0, doa_mod) * noise

            if noise_snr < np.inf:
                for col in ["RF", "PW", "PA", "DOA"]:
                    df.loc[mask, col] = add_noise(df.loc[mask, col], noise_snr)

        if mode == "default":
            df["RF"] += np.random.uniform(*global_rf_shift_range) * noise
            df["PW"] += np.random.uniform(*global_pw_shift_range) * noise
            df["DOA"] += np.random.uniform(*global_doa_shift_range) * noise

    if not reshap and noise_snr < np.inf:
        for i in range(TAG_LEN):
            mask = df["TAG"] == i + 1
            if mask.any():
                for col in ["RF", "PW", "PA", "DOA"]:
                    df.loc[mask, col] = add_noise(df.loc[mask, col], noise_snr)

    if mode == "manual":
        df["PRI"] = df["TOA"].diff().fillna(0)
        d_2["PRI"] = (df["PRI"] - pri_bias) / pri_scale
        d_2["RF"] = mod_normalize(df["RF"], rf_mod)
        d_2["PW"] = mod_normalize(df["PW"], pw_mod)
        d_2["PA"] = normalize(df["PA"])
        d_2["DOA"] = mod_normalize(df["DOA"], doa_mod)
    else:
        if if_pri:
            d_2["PRI"] = df["TOA"].diff().fillna(0) * toa_scale
        else:
            d_2["TOA"] = (df["TOA"] - df["TOA"].min()) * toa_scale

        d_2["RF"] = df["RF"]
        d_2["PW"] = df["PW"] * pw_multiplier
        d_2["PA"] = df["PA"]
        d_2["DOA"] = df["TOA"]

    d_tag["TAG"] = df["TAG"].astype(np.int32)
    return d_2.values, d_tag.values


def _split_pre_reshape_kwargs(kwargs):
    pre_reshape_keys = {
        "sample_ratio",
        "drop_sig_ratio",
        "cut_len",
        "time_scale_mean",
        "time_scale_std",
        "clip_range",
        "time_shift_exp_scale",
    }
    pre_reshape_kwargs = {
        key: value for key, value in kwargs.items() if key in pre_reshape_keys
    }
    to_dict_kwargs = {
        key: value for key, value in kwargs.items() if key not in pre_reshape_keys
    }
    return pre_reshape_kwargs, to_dict_kwargs


def split_label(df: pd.DataFrame) -> List[pd.DataFrame]:
    """分离混杂的 TAG 数据"""
    return [
        df[df["TAG"] == i + 1].copy().reset_index(drop=True) for i in range(TAG_LEN)
    ]


def _gen_mix_job(
    df_list: List[pd.DataFrame],
    size_2,
    if_time_reshap=False,
    test_df_split_list: List[List[pd.DataFrame]] = None,
    t=None,
    **to_dict_kwargs,
):
    """生成混合窗口"""
    if t is None:
        t = range(TAG_LEN)

    if test_df_split_list:
        df_list_new = []
        for i in t:
            candidates = [j for j in [k[i] for k in test_df_split_list] if j.shape[0] > 0] + [df_list[i]]
            df_list_new.append(random.sample(candidates, 1)[0])
    else:
        df_list_new = df_list

    pr_kwargs, dict_kwargs = _split_pre_reshape_kwargs(to_dict_kwargs)

    if if_time_reshap:
        df = pd.concat(
            [pre_reshape(df_list_new[i], **pr_kwargs) for i in t], ignore_index=True
        )
    else:
        df = pd.concat([df_list_new[i] for i in t], ignore_index=True)

    df = df.sort_values(by="TOA", ascending=True, ignore_index=True)

    mixed_windows = []
    for _ in range(size_2):
        if len(df) <= WINDOW_SIZE:
            start = 0
        else:
            start = random.randint(0, len(df) - WINDOW_SIZE)

        m2, m3 = to_dict(df[start : start + WINDOW_SIZE], reshap=True, **dict_kwargs)
        mixed_windows.append([m2, m3])
    return mixed_windows


def mix_data_gen(
    df_list: List[pd.DataFrame],
    size_1: int,
    size_2: int,
    n_jobs: int = -1,
    if_time_reshap=False,
    test_df_split_list=None,
    t=None,
    **to_dict_kwargs,
):
    """并行生成混合窗口"""
    mixed_results = Parallel(n_jobs=n_jobs)(
        delayed(_gen_mix_job)(
            df_list, size_2, if_time_reshap, test_df_split_list, t, **to_dict_kwargs
        )
        for _ in range(size_1)
    )
    mix_windows = []
    for result in mixed_results:
        mix_windows.extend(result)
    return mix_windows


def _target_domain_job(df: pd.DataFrame, size_2, **to_dict_kwargs):
    _, dict_kwargs = _split_pre_reshape_kwargs(to_dict_kwargs)
    mixed_windows = []
    for _ in range(size_2):
        if len(df) <= WINDOW_SIZE:
            start = 0
        else:
            start = random.randint(0, len(df) - WINDOW_SIZE)
        m2, m3 = to_dict(df[start : start + WINDOW_SIZE], reshap=False, **dict_kwargs)
        mixed_windows.append([m2, m3])
    return mixed_windows


def target_domain_data_gen(
    df: pd.DataFrame, size_1: int, size_2: int, n_jobs: int = -1, **to_dict_kwargs
) -> List:
    """生成目标域数据"""
    mixed_results = Parallel(n_jobs=n_jobs)(
        delayed(_target_domain_job)(df, size_2, **to_dict_kwargs) for _ in range(size_1)
    )
    mix_windows = []
    for result in mixed_results:
        mix_windows.extend(result)
    return mix_windows


def make_data(d: List[List[np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成特征和标签张量"""
    inputs = np.array([i[0] for i in d], dtype=np.float32)
    targets = np.array([i[-1] for i in d], dtype=np.int64) - 1
    return torch.FloatTensor(inputs), torch.LongTensor(targets)


class PosMixDatasetCache(Data.Dataset):
    """
    底层数据采样与重生成的 Dataset 缓存封装
    封装了 mix_data_gen / target_domain_data_gen ，自动完成数据并行构建和 Tensor 转换。
    """

    def __init__(
        self,
        df_list,
        size_1=0,
        size_2=0,
        is_test=False,
        is_sequential=False,
        if_mix_test=False,
        test_df_split_list=None,
        n_jobs=-1,
        **to_dict_kwargs,
    ):
        super().__init__()
        self.df_list = df_list
        self.size_1 = size_1
        self.size_2 = size_2
        self.is_test = is_test
        self.is_sequential = is_sequential
        self.if_mix_test = if_mix_test
        self.test_df_split_list = test_df_split_list
        self.n_jobs = n_jobs
        self.to_dict_kwargs = to_dict_kwargs

        self.inputs = None
        self.targets = None
        self.regen_data()

    def regen_data(self):
        _, dict_kwargs = _split_pre_reshape_kwargs(self.to_dict_kwargs)

        if self.is_sequential:
            if not isinstance(self.df_list, list):
                self.df_list = [self.df_list]

            windows = []
            for df in self.df_list:
                for i in range(0, df.shape[0] - WINDOW_SIZE, WINDOW_SIZE):
                    m2, m3 = to_dict(df[i : i + WINDOW_SIZE], reshap=False, **dict_kwargs)
                    windows.append([m2, m3])
        elif self.is_test:
            assert not isinstance(self.df_list, list) or isinstance(
                self.df_list, pd.DataFrame
            ), "is_test assumes a single DataFrame df_list"
            windows = target_domain_data_gen(
                self.df_list,
                self.size_1,
                self.size_2,
                n_jobs=self.n_jobs,
                **self.to_dict_kwargs,
            )
        else:
            t_list = self.test_df_split_list if self.if_mix_test else None
            windows = mix_data_gen(
                self.df_list,
                self.size_1,
                self.size_2,
                n_jobs=self.n_jobs,
                if_time_reshap=True,
                test_df_split_list=t_list,
                **self.to_dict_kwargs,
            )

        self.inputs, self.targets = make_data(windows)

    def __len__(self):
        return len(self.inputs) if self.inputs is not None else 0

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
