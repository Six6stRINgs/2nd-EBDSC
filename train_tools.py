import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Explicit imports
from my_tools import (
    save_checkpoint,
    plot_loss,
    confusion_matrix,
    accuracy_score,
)


def criterion(outputs: torch.FloatTensor, targets: torch.FloatTensor):
    return nn.CrossEntropyLoss()(outputs.view(-1, 12), targets.view(-1))


def evaluate_loader(model, dataloader, device, return_preds=False):
    """通用的单 loader 验证/测试接口，返回 loss 和 acc"""
    model.eval()
    loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            preds = torch.argmax(output.view(-1, 12), dim=-1).detach().cpu().numpy()
            predictions.append(preds)
            targets.append(target.view(-1).detach().cpu().numpy())

            loss += criterion(output, target).item()

    loss /= max(len(dataloader), 1)
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


def train_epoch(model, optimizer, training_loader, device, scaler):
    """单周期训练过程"""
    model.train()
    running_loss = 0.0
    for i, data in enumerate(training_loader):
        inputs, labels = data[0].to(device), data[-1].to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(training_loader)


def evaluate_epoch(model, validing_loader, testing_loader_mini, device):
    """单周期验证过程"""
    v_loss, v_acc = evaluate_loader(model, validing_loader, device)
    t_loss, t_acc = evaluate_loader(model, testing_loader_mini, device)
    return v_loss, t_loss, t_acc


def fit(
    model,
    optimizer,
    lr_scheduler,
    training_loader,
    validing_loader,
    testing_loader_mini,
    parser_args,
    device,
    NAME,
):
    """标准化训练管线编排 (Epoch 循环)"""
    scaler = GradScaler()
    loss_record = {"train": [], "vaild": [], "test": [], "acc": []}
    epoch_start = 0
    test_loss_min = 2.0

    print(
        "参数量：",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        end=" ",
    )
    print(NAME, "start train:", epoch_start)

    l = tqdm(
        range(epoch_start, epoch_start + parser_args.max_epoch), dynamic_ncols=True
    )
    for epoch in l:
        # 重生成数据集
        if epoch % parser_args.rg == 0 and epoch != 0:
            training_loader.dataset.hard = parser_args.hard * 0.01
            training_loader.dataset.base_dataset.regen_data()
            validing_loader.dataset.base_dataset.regen_data()

        # [Eval Step]
        v, t, a = evaluate_epoch(model, validing_loader, testing_loader_mini, device)
        loss_record["vaild"].append(v)
        loss_record["test"].append(t)
        loss_record["acc"].append(a)
        plot_loss(loss_record, f"{NAME}_{now}")

        # 保存最优模型
        if test_loss_min > loss_record["test"][-1]:
            test_loss_min = loss_record["test"][-1]
            print(
                f'Epoch {epoch} test loss min {test_loss_min}, acc {loss_record["acc"][-1]}'
            )
            save_checkpoint(
                epoch,
                loss_record,
                model,
                optimizer,
                f"./saved_models/{NAME}_{now}_mloss.pth",
            )

        # [Train Step]
        train_loss = train_epoch(model, optimizer, training_loader, device, scaler)

        lr_scheduler.step()
        loss_record["train"].append(train_loss)

        l.set_description(
            "E%d, loss=%.3f, vaild=%.3f, test=%.3f, acc=%.2f"
            % (
                epoch + 1,
                train_loss,
                loss_record["vaild"][-1],
                loss_record["test"][-1],
                loss_record["acc"][-1],
            )
        )

        if epoch % 50 == 49:
            save_checkpoint(
                epoch, loss_record, model, optimizer, f"./saved_models/{NAME}_{now}.pth"
            )

    print("Finished Training")
    save_checkpoint(
        epoch,
        loss_record,
        model,
        optimizer,
        f"./saved_models/{NAME}_{now}_cp-{epoch}.pth",
    )
    plot_loss(loss_record, f"{NAME}_{now}")
    return loss_record


def evaluate_loader(model, dataloader, device, criterion, return_preds=False):
    """通用的单 loader 验证/测试接口，返回 loss 和 acc (以及 可选的 predictions, targets)"""
    model.eval()
    loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            preds = torch.argmax(output.view(-1, 12), dim=-1).detach().cpu().numpy()
            predictions.append(preds)
            targets.append(target.view(-1).detach().cpu().numpy())

            loss += criterion(output, target).item()

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


def test_final(model, loaders_dict, device, NAME):
    """训练完成后的最终验证过程"""
    checkpoint = torch.load(f"./saved_models/{NAME}_{now}_mloss.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    for name, loader in loaders_dict.items():
        loss, acc, preds, targets = evaluate_loader(
            model, loader, device, return_preds=True
        )
        print(f"[{name}] Average loss: {loss:.4f}, Acc: {acc:.4f}")
        confusion_matrix(preds, targets, f"{NAME}_{now}_{name}_mloss{loss:.3f}")
