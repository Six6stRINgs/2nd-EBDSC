import torch
import os
from torch.cuda.amp import GradScaler
import datetime
import argparse
from tqdm import tqdm

from train_tools import train_epoch, evaluate_loader, test_final
from my_tools import (
    seed_everything,
    save_checkpoint,
    plot_loss,
)

from data import build_dataloaders
from models import build_model

os.environ["OMP_NUM_THREADS"] = "1"

# Constants
TAG_LEN = 12
INPUT_CHANNELS = 5
now = datetime.datetime.now().strftime("%m%d_%H-%M")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Code for 2nd EBDSC -- Wide-Value-Embs TCN -- by framist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cuda", type=int, default=0, help="所使用的 cuda 设备，暂不支持多设备并行"
    )
    parser.add_argument(
        "--num_layers", type=int, default=24, help="layers of modernTCN"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--ratio", type=int, default=2, help="ffn ratio")
    parser.add_argument("--ls", type=int, default=51, help="large kernel sizes")
    parser.add_argument("--ss", type=int, default=5, help="small kernel size")
    parser.add_argument("--dp", type=float, default=0.5, help="drop out")
    parser.add_argument("--hard", type=int, default=80, help="hard ratio (%) for mask")
    parser.add_argument("--rg", type=int, default=999, help="re-gen data epoch")
    parser.add_argument("--max_epoch", type=int, default=400, help="max train epoch")
    parser.add_argument(
        "--mix_test", action="store_true", default=False, help="是否混入测试集训练"
    )

    # 对照、消融实验的一些参数
    parser.add_argument(
        "--learnable_emb",
        action="store_true",
        default=False,
        help="是否使用可学习的 emb",
    )
    parser.add_argument(
        "--model", type=str, default="ModernTCN", help="backbone 模型选择"
    )
    parser.add_argument(
        "--manual", action="store_true", default=False, help="是否手动构建交织"
    )
    parser.add_argument(
        "--pri", action="store_true", default=False, help="是否使用 PRI 而非 TOA"
    )
    parser.add_argument(
        "--fmask",
        type=str,
        default="r",
        help="mask 方式 r: randMask, m: meanMask, c: constMask",
    )

    return parser.parse_args()


def main():
    seed_everything()
    parser_args = parse_args()

    use_cuda = True
    device = torch.device(
        f"cuda:{parser_args.cuda}"
        if (use_cuda and torch.cuda.is_available())
        else "cpu"
    )
    print("CUDA Available: ", torch.cuda.is_available(), "use:", device)

    to_dict_params = {}
    if parser_args.manual and parser_args.learnable_emb:
        to_dict_params["mode"] = "manual"
        DATA_NAME = "PosAllManualInterleavePRI"
    elif parser_args.pri:
        to_dict_params["if_pri"] = True
        DATA_NAME = "PosAllPRI"
    else:
        DATA_NAME = "PosAll"

    if parser_args.fmask == "r":
        F_MASK = lambda x: torch.rand_like(x) * 2 - 1
    elif parser_args.fmask == "m":
        F_MASK = lambda x: torch.mean(x, axis=0)
    elif parser_args.fmask == "c":
        F_MASK = lambda x: torch.zeros_like(x)

    NAME = f"{DATA_NAME}{parser_args.hard}Hr{parser_args.fmask}{parser_args.rg}R"
    if parser_args.mix_test:
        NAME = NAME + "_MT"

    if parser_args.learnable_emb:
        NAME = NAME + "_LEmb"
        from data.my_datastes import MyDataSet_woEmb as MyDataSet
    else:
        from data.my_datastes import MyDataSet

    model, optimizer, lr_scheduler, NAME, learn_rate = build_model(
        parser_args, device, NAME
    )

    training_loader, validing_loader, testing_loader, testing_loader_mini = (
        build_dataloaders(parser_args, to_dict_params, F_MASK, MyDataSet)
    )

    # --- 训练核心逻辑 (取代 fit) ---
    scaler = GradScaler()
    loss_record = {"train": [], "vaild": [], "test": [], "acc": []}
    test_loss_min = 10.0
    epoch_start = 0

    print(
        f"参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)} 层数：{parser_args.num_layers}"
    )
    print(f"{NAME} start train at {now}")

    for epoch in range(epoch_start, epoch_start + parser_args.max_epoch):
        train_loss = train_epoch(
            model, optimizer, training_loader, device, scaler, epoch
        )

        lr_scheduler.step()
        loss_record["train"].append(train_loss)

        if epoch % parser_args.rg == 0 and epoch != 0:
            training_loader.dataset.hard = parser_args.hard * 0.01
            training_loader.dataset.base_dataset.regen_data()
            validing_loader.dataset.base_dataset.regen_data()

        v_loss, t_loss, t_acc = evaluate_loader(
            model, validing_loader, testing_loader_mini, device
        )
        loss_record["vaild"].append(v_loss)
        loss_record["test"].append(t_loss)
        loss_record["acc"].append(t_acc)
        plot_loss(loss_record, f"{NAME}_{now}")

        if test_loss_min > t_loss:
            test_loss_min = t_loss
            print(
                f"Epoch {epoch} best test loss: {test_loss_min:.4f}, acc: {t_acc:.4f}"
            )
            save_checkpoint(
                epoch,
                loss_record,
                model,
                optimizer,
                f"./saved_models/{NAME}_{now}_mloss.pth",
            )

        print(
            f"Epoch {epoch+1}/{parser_args.max_epoch} - loss: {train_loss:.4f}, v_loss: {v_loss:.4f}, t_mini_loss: {t_loss:.4f}, t_mini_acc: {t_acc:.4f}"
        )

        # 5. 定期保存
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

    # --- 最终评估 ---
    print(f"learn_rate={learn_rate} batch_size={parser_args.batch_size}")
    if loss_record["train"]:
        print(
            f"Final Train Loss: {loss_record['train'][-1]:.4f}, Valid Loss: {v_loss:.4f}, Test Mini Loss: {t_loss:.4f}"
        )

    loaders_dict = {
        "valid": validing_loader,
        "test_mini": testing_loader_mini,
        "test_full": testing_loader,
    }
    test_final(model, loaders_dict, device, NAME, now)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
