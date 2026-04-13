import argparse
import datetime
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch.cuda.amp import GradScaler

from utils.config_utils import apply_cli_overrides, load_config, to_namespace
from data import build_dataloaders, configure_data_globals, get_data_globals
from models import build_model
from my_tools import TrainingLogger, plot_loss, save_checkpoint, seed_everything
from train_tools import criterion, evaluate_epoch, test_final, train_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code for 2nd EBDSC -- Wide-Value-Embs TCN -- by framist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="数据集根目录，覆盖 config.yaml"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="数据名称，覆盖 config.yaml 中的 experiment.data_name",
    )
    parser.add_argument(
        "--cuda", type=int, default=None, help="所使用的 cuda 设备，覆盖 config.yaml"
    )
    parser.add_argument(
        "--model",
        "--model_name",
        dest="model_name",
        type=str,
        default="ModernTCN",
        help="backbone 模型选择，覆盖 config.yaml",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="batch size，覆盖 config.yaml"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate，覆盖 config.yaml"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=50, help="max train epoch，覆盖 config.yaml"
    )
    parser.add_argument(
        "--seed", type=int, default=3407, help="random seed，覆盖 config.yaml"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader，覆盖 config.yaml",
    )
    parser.add_argument(
        "--mix_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否混入测试集训练，覆盖 config.yaml",
    )
    parser.add_argument(
        "--learnable_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用可学习 emb，覆盖 config.yaml",
    )
    parser.add_argument(
        "--manual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否手动构建交织，覆盖 config.yaml",
    )
    parser.add_argument(
        "--pri",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用 PRI 而非 TOA，覆盖 config.yaml",
    )

    return parser.parse_args()


def resolve_data_name(config):
    to_dict_params = {}
    data_name = "PosAll"

    if config.experiment.manual and config.experiment.learnable_emb:
        to_dict_params["mode"] = "manual"
        data_name = "PosAllManualInterleavePRI"
    elif config.experiment.pri:
        to_dict_params["if_pri"] = True
        data_name = "PosAllPRI"

    if config.experiment.data_name is not None:
        data_name = config.experiment.data_name

    return data_name, to_dict_params


def build_f_mask(strategy: str):
    if strategy == "r":

        def f_mask(x):
            return torch.rand_like(x) * 2 - 1

    elif strategy == "m":

        def f_mask(x):
            return torch.mean(x, axis=0)

    elif strategy == "c":

        def f_mask(x):
            return torch.zeros_like(x)

    else:
        raise ValueError(f"Unsupported masking strategy: {strategy}")

    return f_mask


def main():
    cli_args = parse_args()
    raw_config = load_config(cli_args.config)
    merged_config, applied_overrides = apply_cli_overrides(raw_config, cli_args)
    config = to_namespace(merged_config)

    os.environ["OMP_NUM_THREADS"] = str(config.runtime.omp_num_threads)
    seed_everything(config.runtime.seed)

    now = datetime.datetime.now().strftime("%m%d_%H-%M")
    Path(config.paths.saved_models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.saved_figs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.logs_dir).mkdir(parents=True, exist_ok=True)

    use_cuda = True
    device = torch.device(
        f"cuda:{config.runtime.cuda}"
        if (use_cuda and torch.cuda.is_available())
        else "cpu"
    )
    amp_enabled = bool(config.runtime.use_amp and device.type == "cuda")
    print("CUDA Available: ", torch.cuda.is_available(), "use:", device)
    if applied_overrides:
        print(f"CLI overrides: {applied_overrides}")

    data_name, to_dict_params = resolve_data_name(config)
    configure_data_globals(
        tag_len=config.task.tag_len,
        window_size=config.task.window_size,
        dataset_dir=config.paths.dataset_dir,
        data_name=data_name,
    )
    data_config = get_data_globals()
    f_mask = build_f_mask(config.masking.strategy)

    name = (
        f"{data_config['DATA_NAME']}{config.masking.hard_ratio}Hr"
        f"{config.masking.strategy}{config.train.regen_interval}R"
    )
    if config.experiment.mix_test:
        name = name + "_MT"

    if config.experiment.learnable_emb:
        name = name + "_LEmb"
        from data.dataset import ExtraDataset_woEmb as ExtraDataset
    else:
        from data.dataset import ExtraDataset

    model, optimizer, lr_scheduler, name, learn_rate = build_model(config, device, name)
    log_path = Path(config.paths.logs_dir) / f"{name}_{now}.log"
    train_logger = TrainingLogger(str(log_path))

    training_loader, validing_loader, testing_loader, testing_loader_mini = (
        build_dataloaders(config, to_dict_params, f_mask, ExtraDataset)
    )

    scaler = GradScaler(enabled=amp_enabled)
    loss_record = {"train": [], "vaild": [], "test": [], "acc": [], "v_acc": []}
    test_loss_min = float("inf")
    epoch_start = 0

    model_summary = f"参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)} 层数：{config.model.num_layers}"
    print(model_summary)
    print(f"数据配置：{data_config}")
    print(f"{name} start train at {now}")
    train_logger.section("Run")
    train_logger.log(f"name={name}")
    train_logger.log(f"start_time={now}")
    train_logger.log(f"device={device}")
    train_logger.log(model_summary)
    train_logger.log(f"data_config={data_config}")
    if applied_overrides:
        train_logger.log(f"cli_overrides={applied_overrides}")

    for epoch in range(epoch_start, epoch_start + config.train.max_epoch):
        train_loss = train_epoch(
            model,
            optimizer,
            training_loader,
            device,
            scaler,
            epoch,
            num_classes=config.task.num_classes,
            amp_enabled=amp_enabled,
        )

        lr_scheduler.step()
        loss_record["train"].append(train_loss)

        if (
            config.train.regen_interval > 0
            and epoch % config.train.regen_interval == 0
            and epoch != 0
        ):
            training_loader.dataset.hard = config.masking.hard_ratio * 0.01
            training_loader.dataset.base_dataset.regen_data()
            validing_loader.dataset.base_dataset.regen_data()

        v_loss, v_acc = evaluate_epoch(
            model,
            validing_loader,
            device,
            criterion,
            return_preds=False,
            desc="Valid",
            num_classes=config.task.num_classes,
        )

        t_loss, t_acc = evaluate_epoch(
            model,
            testing_loader_mini,
            device,
            criterion,
            return_preds=False,
            desc="Test Mini",
            num_classes=config.task.num_classes,
        )

        loss_record["vaild"].append(v_loss)
        loss_record["test"].append(t_loss)
        loss_record["v_acc"].append(v_acc)
        loss_record["acc"].append(t_acc)
        plot_loss(loss_record, f"{name}_{now}", output_dir=config.paths.saved_figs_dir)

        if config.train.best_ckpt_metric == "test_loss" and test_loss_min > t_loss:
            test_loss_min = t_loss
            best_line = (
                f"Epoch {epoch} best test loss: {test_loss_min:.4f}, acc: {t_acc:.4f}"
            )
            print(best_line)
            train_logger.log(best_line)
            save_checkpoint(
                epoch,
                loss_record,
                model,
                optimizer,
                str(Path(config.paths.saved_models_dir) / f"{name}_{now}_mloss.pth"),
            )

        epoch_line = f"Epoch {epoch+1}/{config.train.max_epoch} - loss: {train_loss:.4f}, v_loss: {v_loss:.4f}, v_acc: {v_acc:.4f}, t_mini_loss: {t_loss:.4f}, t_mini_acc: {t_acc:.4f}"
        print(epoch_line)
        train_logger.log(epoch_line)

        if (
            config.train.save_every > 0
            and epoch % config.train.save_every == config.train.save_every - 1
        ):
            save_checkpoint(
                epoch,
                loss_record,
                model,
                optimizer,
                str(Path(config.paths.saved_models_dir) / f"{name}_{now}.pth"),
            )

    print("Finished Training")
    train_logger.log("Finished Training")

    last_epoch_weight = str(Path(config.paths.saved_models_dir) / f"{name}_{now}.pth")

    save_checkpoint(
        epoch,
        loss_record,
        model,
        optimizer,
        str(last_epoch_weight),
    )
    plot_loss(loss_record, f"{name}_{now}", output_dir=config.paths.saved_figs_dir)

    final_line = f"learn_rate={learn_rate} batch_size={config.train.batch_size}"
    print(final_line)
    train_logger.log(final_line)
    if loss_record["train"]:
        summary_line = f"Final Train Loss: {loss_record['train'][-1]:.4f}, Valid Loss: {v_loss:.4f}, Test Mini Loss: {t_loss:.4f}"
        print(summary_line)
        train_logger.log(summary_line)

    loaders_dict = {
        "valid": validing_loader,
        "test_mini": testing_loader_mini,
        "test_full": testing_loader,
    }
    test_final(
        model,
        loaders_dict,
        last_epoch_weight,
        device,
        name,
        now,
        num_classes=config.task.num_classes,
        tag_len=config.task.tag_len,
        figures_dir=config.paths.saved_figs_dir,
        checkpoint_dir=config.paths.saved_models_dir,
        logger=train_logger,
    )
    train_logger.log(f"log_path={log_path}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
