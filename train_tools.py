import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from my_tools import accuracy_score, confusion_matrix


def criterion(outputs: torch.FloatTensor, targets: torch.FloatTensor, num_classes: int = 12):
    return nn.CrossEntropyLoss()(outputs.view(-1, num_classes), targets.view(-1))


def train_epoch(
    model,
    optimizer,
    training_loader,
    device,
    scaler,
    epoch_idx,
    num_classes: int = 12,
    amp_enabled: bool = True,
):
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

        with autocast(enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels, num_classes=num_classes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(training_loader)


def evaluate_epoch(
    model,
    dataloader,
    device,
    criterion_fn,
    return_preds=False,
    desc=None,
    num_classes: int = 12,
):
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
            desc=f"Evaluate {desc}" if desc else "Evaluate",
            leave=False,
            dynamic_ncols=True,
        )
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[-1].to(device)
            output = model(inputs)

            batch_preds_raw = torch.argmax(output.view(-1, num_classes), dim=-1)
            batch_labels_raw = labels.view(-1)
            correct_total += (batch_preds_raw == batch_labels_raw).sum().item()
            sample_total += batch_labels_raw.numel()

            preds = batch_preds_raw.detach().cpu().numpy()
            predictions.append(preds)
            targets.append(batch_labels_raw.detach().cpu().numpy())

            batch_loss = criterion_fn(output, labels, num_classes=num_classes).item()
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


def test_final(
    model,
    loaders_dict,
    path,
    device,
    name,
    now,
    num_classes: int = 12,
    tag_len: int = 12,
    figures_dir: str = "saved_figs",
    checkpoint_dir: str = "saved_models",
    logger=None,
):
    """训练完成后的最终验证过程"""
    load_line = f"Loading best model from {path} for final evaluation..."
    print(load_line)
    if logger is not None:
        logger.log(load_line)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    for loader_name, loader in loaders_dict.items():
        loss, acc, preds, targets = evaluate_epoch(
            model,
            loader,
            device,
            criterion,
            return_preds=True,
            num_classes=num_classes,
        )
        final_eval_line = f"[{loader_name}] Final Average loss: {loss:.4f}, Acc: {acc:.4f}"
        print(final_eval_line)
        if logger is not None:
            logger.log(final_eval_line)
        confusion_matrix(
            preds,
            targets,
            f"{name}_{now}_{loader_name}_mloss{loss:.3f}",
            tag_len=tag_len,
            output_dir=figures_dir,
        )
