import numpy as np
import pandas as pd
from typing import List, Tuple
from joblib import Parallel, delayed
import random
import torch
import torch.utils.data as Data

# 全局常量
TAG_LEN = 12
WINDOW_SIZE = 1024
DATASET_DIR = "/root/autodl-tmp/datasets/extra"
DATA_NAME = "PosAll"  # 默认名称，可在调用处动态修改或通过参数传递


def read_txt(filepath):
    """读取 txt 文件，并以 DataFrame 返回"""
    columns = ["TOA", "RF", "PW", "PA", "DOA", "TAG"]
    df = pd.read_csv(filepath, sep=r"\s+", names=columns)
    # 数据类型规范
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
    clip_range: Tuple[float, float] = (0.5, 2.0),
) -> pd.DataFrame:
    """数据时间尺度上变化，模拟频移等增强"""
    cut_len = 5
    start = np.random.uniform(df["TOA"].min(), df["TOA"].max())
    df_scaled: pd.DataFrame = df[
        (df["TOA"] >= start) & (df["TOA"] <= start + cut_len)
    ].copy()

    # 随机信号丢失
    if sample_ratio is None:
        df_scaled = df_scaled.sample(frac=np.random.uniform(drop_sig_ratio, 1))
    else:
        df_scaled = df_scaled.sample(frac=sample_ratio)

    # 频移/比例缩放
    df_scaled["TOA"] *= np.clip(np.random.normal(1, 0.2), *clip_range)

    # 去除原始的 TOA 偏移
    df_scaled["TOA"] -= df_scaled["TOA"].min()

    # 随机进入时移
    df_scaled["TOA"] += np.clip(np.random.exponential(1), 0, cut_len)

    return df_scaled


def to_dict(
    df: pd.DataFrame,
    reshap=False,
    mode="default",
    noise=1.0,
    noise_snr=np.inf,
    if_pri=False,
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

    df = df.copy()
    d_2 = pd.DataFrame()
    d_tag = pd.DataFrame()

    DOA_MOD, RF_MOD, PW_MOD = 90, 100, 10

    if reshap:
        for i in range(TAG_LEN):
            mask = df["TAG"] == i + 1
            if not mask.any():
                continue

            # RF 变换
            di = df.loc[mask, "RF"]
            scale = np.random.uniform(0.2, 2) * noise
            offset = (
                np.random.uniform(0, RF_MOD)
                if mode == "manual"
                else np.random.normal(0, 200)
            ) * noise
            df.loc[mask, "RF"] = scale * (di - di.mean()) + di.mean() + offset

            # PW 变换
            di = df.loc[mask, "PW"]
            scale = np.random.uniform(0.5, 2) * noise
            offset = (
                np.random.uniform(0, PW_MOD)
                if mode == "manual"
                else np.random.normal(0, 10)
            ) * noise
            df.loc[mask, "PW"] = scale * (di - di.mean()) + di.mean() + offset

            # DOA 变换
            df.loc[mask, "DOA"] += np.random.uniform(0, DOA_MOD) * noise

            # 增加高斯噪声
            if noise_snr < np.inf:
                for col in ["RF", "PW", "PA", "DOA"]:
                    df.loc[mask, col] = add_noise(df.loc[mask, col], noise_snr)

        # 整体偏移
        if mode == "default":
            df["RF"] += np.random.uniform(-1000, 1000) * noise
            df["PW"] += np.random.uniform(-10, 10) * noise
            df["DOA"] += np.random.uniform(-60, 60) * noise

    # 处理噪声 (不需要 reshap==True 也可以加)
    if not reshap and noise_snr < np.inf:
        for i in range(TAG_LEN):
            mask = df["TAG"] == i + 1
            if mask.any():
                for col in ["RF", "PW", "PA", "DOA"]:
                    df.loc[mask, col] = add_noise(df.loc[mask, col], noise_snr)

    # 特征工程与归一化
    if mode == "manual":
        df["PRI"] = df["TOA"].diff().fillna(0)
        d_2["PRI"] = (df["PRI"] - 0.0002) / 0.0005
        d_2["RF"] = mod_normalize(df["RF"], RF_MOD)
        d_2["PW"] = mod_normalize(df["PW"], PW_MOD)
        d_2["PA"] = normalize(df["PA"])
        d_2["DOA"] = mod_normalize(df["DOA"], DOA_MOD)
    else:
        # Default mode (air, PRI, all)
        if if_pri:
            d_2["PRI"] = df["TOA"].diff().fillna(0) * 5e5
        else:
            d_2["TOA"] = (df["TOA"] - df["TOA"].min()) * 5e5

        d_2["RF"] = df["RF"]
        d_2["PW"] = df["PW"] * 10
        d_2["PA"] = df["PA"]
        d_2["DOA"] = df["TOA"]  # 原始代码中 DOA 赋值为 TOA，保持一致

    d_tag["TAG"] = df["TAG"].astype(np.int32)
    return d_2.values, d_tag.values


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
            candidates = [
                j for j in [k[i] for k in test_df_split_list] if j.shape[0] > 0
            ] + [df_list[i]]
            df_list_new.append(random.sample(candidates, 1)[0])
    else:
        df_list_new = df_list

    # pre_reshape 参数传递可以通过 to_dict_kwargs 提取或另设，这里暂保持现状
    # 提取 pre_reshape 相关参数
    pr_kwargs = {}
    if "sample_ratio" in to_dict_kwargs:
        pr_kwargs["sample_ratio"] = to_dict_kwargs.pop("sample_ratio")
    if "drop_sig_ratio" in to_dict_kwargs:
        pr_kwargs["drop_sig_ratio"] = to_dict_kwargs.pop("drop_sig_ratio")
    if "clip_range" in to_dict_kwargs:
        pr_kwargs["clip_range"] = to_dict_kwargs.pop("clip_range")

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

        m2, m3 = to_dict(df[start : start + WINDOW_SIZE], reshap=True, **to_dict_kwargs)
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
    mixed_windows = []
    for _ in range(size_2):
        if len(df) <= WINDOW_SIZE:
            start = 0
        else:
            start = random.randint(0, len(df) - WINDOW_SIZE)
        m2, m3 = to_dict(
            df[start : start + WINDOW_SIZE], reshap=False, **to_dict_kwargs
        )
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
    def __init__(self, df_list, size_1=0, size_2=0, is_test=False, is_sequential=False, if_mix_test=False, test_df_split_list=None, **to_dict_kwargs):
        super().__init__()
        self.df_list = df_list
        self.size_1 = size_1
        self.size_2 = size_2
        self.is_test = is_test
        self.is_sequential = is_sequential
        self.if_mix_test = if_mix_test
        self.test_df_split_list = test_df_split_list
        self.to_dict_kwargs = to_dict_kwargs
        
        self.inputs = None
        self.targets = None
        self.regen_data()

    def regen_data(self):
        if self.is_sequential:
            # 顺序生成测试数据窗 (针对完整的 df_list 进行滑动窗切片)
            windows = []
            for df in self.df_list:
                for i in range(0, df.shape[0] - WINDOW_SIZE, WINDOW_SIZE):
                    m2, m3 = to_dict(df[i:i+WINDOW_SIZE], reshap=False, **self.to_dict_kwargs)
                    windows.append([m2, m3])
        elif self.is_test:
            # 目标域抽样测试数据
            assert not isinstance(self.df_list, list) or isinstance(self.df_list, pd.DataFrame), "is_test assumes a single DataFrame df_list"
            windows = target_domain_data_gen(self.df_list, self.size_1, self.size_2, **self.to_dict_kwargs)
        else:
            # 训练/验证随机抽样打乱数据
            t_list = self.test_df_split_list if self.if_mix_test else None
            windows = mix_data_gen(self.df_list, self.size_1, self.size_2, n_jobs=-1, if_time_reshap=True, test_df_split_list=t_list, **self.to_dict_kwargs)
        
        self.inputs, self.targets = make_data(windows)

    def __len__(self):
        return len(self.inputs) if self.inputs is not None else 0
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
