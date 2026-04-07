import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# Explicit imports
from my_tools import (
    confusion_matrix,
    accuracy_score,
)


def criterion(outputs: torch.FloatTensor, targets: torch.FloatTensor):
    return nn.CrossEntropyLoss()(outputs.view(-1, 12), targets.view(-1))


def train_epoch(model, optimizer, training_loader, device, scaler, epoch_idx):
    """单周期训练过程，包含 Batch 级别的 tqdm 进度条"""
    model.train()
    running_loss = 0.0
    pbar = tqdm(
        enumerate(training_loader),
        total=len(training_loader),
        desc=f"Epoch {epoch_idx}",
        leave=False,
        dynamic_ncols=True,
    )
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[-1].to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(training_loader)


def evaluate_epoch(model, dataloader, device, criterion, return_preds=False):
    """通用的单 loader 验证/测试接口，返回 loss 和 acc (以及 可选的 predictions, targets)"""
    model.eval()
    loss = 0.0
    correct_total = 0
    sample_total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Evaluate",
            leave=False,
            dynamic_ncols=True,
        )
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[-1].to(device)
            output = model(inputs)

            # 计算准确率
            batch_preds_raw = torch.argmax(output.view(-1, 12), dim=-1)
            batch_labels_raw = labels.view(-1)
            correct_total += (batch_preds_raw == batch_labels_raw).sum().item()
            sample_total += batch_labels_raw.numel()

            # 存储以进行最终统计
            preds = batch_preds_raw.detach().cpu().numpy()
            predictions.append(preds)
            targets.append(batch_labels_raw.detach().cpu().numpy())

            batch_loss = criterion(output, labels).item()
            loss += batch_loss

            pbar.set_postfix(
                {
                    "avg_loss": f"{loss / (i + 1):.4f}",
                    "acc": f"{correct_total / sample_total:.4f}",
                }
            )

    loss /= len(dataloader)
    if not predictions:
        if return_preds:
            return loss, 0.0, np.array([]), np.array([])
        return loss, 0.0

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, predictions)

    if return_preds:
        return loss, acc, predictions, targets
    return loss, acc


def evaluate_loader(model, validing_loader, testing_loader_mini, device):
    """单周期验证过程"""
    v_loss, v_acc = evaluate_epoch(model, validing_loader, device, criterion)
    t_loss, t_acc = evaluate_epoch(model, testing_loader_mini, device, criterion)
    return v_loss, t_loss, t_acc


def test_final(model, loaders_dict, device, NAME, now):
    """训练完成后的最终验证过程"""
    path = f"./saved_models/{NAME}_{now}_mloss.pth"
    print(f"Loading best model from {path} for final evaluation...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    for name, loader in loaders_dict.items():
        loss, acc, preds, targets = evaluate_epoch(
            model, loader, device, criterion, return_preds=True
        )
        print(f"[{name}] Final Average loss: {loss:.4f}, Acc: {acc:.4f}")
        confusion_matrix(preds, targets, f"{NAME}_{now}_{name}_mloss{loss:.3f}")
