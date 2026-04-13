from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def seed_everything(seed: int = 3407):
    """ref. torch.manual_seed(3407) is all you need"""
    import os
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"set all seed: {seed}")


class TrainingLogger:
    """Write concise training summaries to a log file without recording tqdm output."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        line = message.rstrip()
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def section(self, title: str):
        self.log(f"[{title}]")


def TSNE_visualization(
    s_feature: np.ndarray,
    s_labels: np.ndarray,
    t_feature: np.ndarray,
    t_labels: np.ndarray,
    tag_len: int = 12,
):
    """TSNE 可视化域间的分布
    输入都是 2 维 numpy 数组，维度 1: 样本，维度 2: 特征
    s: 源域
    t: 目标域"""
    tsne = TSNE(n_components=2)

    feature = np.concatenate((s_feature, t_feature), axis=0)
    feature = tsne.fit_transform(feature)
    cut = s_feature.shape[0]
    s_feature = feature[:cut, :]
    t_feature = feature[cut:, :]

    plt.figure()
    plt.scatter(feature[:cut, 0], feature[:cut, 1], c="r", label="源域", s=1, alpha=0.2)
    plt.scatter(feature[cut:, 0], feature[cut:, 1], c="b", label="目标域", s=1, alpha=0.2)
    plt.legend()
    plt.plot()

    s_feature = feature[:cut, :]
    plt.figure()
    plt.scatter(feature[cut:, 0], feature[cut:, 1], c="gray", label="目标域", s=1, alpha=0.1)
    for i in range(0, tag_len):
        plt.scatter(s_feature[s_labels == i, 0], s_feature[s_labels == i, 1], label=i + 1, s=1, alpha=0.7)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("源域 t-SNE Visualization")
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(feature[:cut, 0], feature[:cut, 1], c="gray", label="源域", s=1, alpha=0.1)
    for i in range(0, tag_len):
        plt.scatter(t_feature[t_labels == i, 0], t_feature[t_labels == i, 1], label=i + 1, s=1, alpha=0.7)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("目标域 t-SNE Visualization")
    plt.legend()
    plt.show()


def save_checkpoint(epoch, loss_record, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path):
    """保存模型 checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "loss_record": loss_record,
        "model": model,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(
    model,
    path,
    optimizer: torch.optim.Optimizer = None,
    device="cuda",
) -> Tuple[int, Dict[str, List[float]]]:
    """加载 checkpoint"""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    loss_record = state["loss_record"]
    return state["epoch"], loss_record


def accuracy(predictions: np.ndarray, targets: np.ndarray):
    """计算硬标签的准确率"""
    return np.mean(predictions == targets)


def acc_logit(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 12):
    """计算 logits 的准确率"""
    predictions = np.argmax(logits.view(-1, num_classes).detach().cpu().numpy(), axis=1) + 1
    targets = np.argmax(targets.view(-1, num_classes).detach().cpu().numpy(), axis=1) + 1
    return accuracy(predictions, targets)


def confusion_matrix(
    predictions,
    targets,
    plot_name: str = None,
    tag_len: int = 12,
    average: str = "weighted",
    if_save=True,
    output_dir: str = "saved_figs",
):
    """混淆矩阵"""
    matrix = np.zeros((tag_len, tag_len))

    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1

    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    f1 = f1_score(targets, predictions, average=average)
    log = f"Acc: {acc*100:.3f}, Pre: {precision*100:.3f}, Rec: {recall*100:.3f}, F1: {f1*100:.3f} [{average}]"
    print(plot_name)
    print(log)

    if plot_name:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.imshow(matrix / np.maximum(1, np.sum(matrix, axis=1)[:, None]))
        for i in range(tag_len):
            for j in range(tag_len):
                plt.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="blue")
        plt.title("confusion matrix")
        plt.xlabel("prediction")
        plt.ylabel("target")
        plt.colorbar()
        plt.title(f"{plot_name}\n{log}")
        if if_save:
            plt.savefig(Path(output_dir) / f"{plot_name}.png")
        plt.show()

    return acc


def plot_loss(loss_record: Dict[str, List[float]], plot_name: str, output_dir: str = "saved_figs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(loss_record["train"], label="train", linestyle="-", marker=".", linewidth=1, alpha=0.6)
    plt.plot(loss_record["vaild"], label="vaild", linestyle="-", marker=".", linewidth=1, alpha=0.6)
    plt.plot(loss_record["test"], label="test", alpha=0.9)
    plt.plot(loss_record["acc"], label="acc", alpha=0.9)
    plt.grid()
    plt.ylim(0, 3)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f'{plot_name}\nmax acc: {max(loss_record["acc"]):.3f} min loss:{min(loss_record["test"]):.5f}')
    plt.savefig(Path(output_dir) / f"{plot_name}_loss.png")
    plt.show()
    plt.close()


def load_pretrained_params(
    model: torch.nn.Module, path="./my_models/tf_s_2time5class_1000_minloss_cp-941.pth"
):
    """加载预训练参数，只加载同名的层"""
    pretrained_dict = torch.load(path)["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
