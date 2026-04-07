import torch
import torch.utils.data as Data
import datetime
import argparse
from train_tools import fit, test_final

# Explicit imports
from mix_data_pos import read_dfs, split_label, PosMixDatasetCache
from my_tools import (
    seed_everything,
)

import os

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
        "--model", type=str, default="modernTCN", help="backbone 模型选择"
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


def build_dataloaders(parser_args, to_dict_params, F_MASK, MyDataSet):
    df_list, test_df_list = read_dfs()
    test_df_split_list = (
        [split_label(i) for i in test_df_list] if parser_args.mix_test else None
    )

    # Train / Valid Base
    d_train_base = PosMixDatasetCache(
        df_list,
        100,
        100,
        is_test=False,
        if_mix_test=parser_args.mix_test,
        test_df_split_list=test_df_split_list,
        **to_dict_params,
    )
    d_valid_base = PosMixDatasetCache(
        df_list,
        20,
        50,
        is_test=False,
        if_mix_test=parser_args.mix_test,
        test_df_split_list=test_df_split_list,
        **to_dict_params,
    )

    training_loader = Data.DataLoader(
        MyDataSet(d_train_base, hard=parser_args.hard * 0.01, f_mask=F_MASK),
        batch_size=parser_args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validing_loader = Data.DataLoader(
        MyDataSet(d_valid_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Test Loaders
    d_test_base = PosMixDatasetCache(test_df_list, is_sequential=True, **to_dict_params)
    testing_loader = Data.DataLoader(
        MyDataSet(d_test_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
    )

    d_test_mini_base = PosMixDatasetCache(
        test_df_list[2], 20, 50, is_test=True, **to_dict_params
    )
    testing_loader_mini = Data.DataLoader(
        MyDataSet(d_test_mini_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
    )

    return training_loader, validing_loader, testing_loader, testing_loader_mini


def build_model(parser_args, device, NAME):
    D = 128
    learning_rate = 4e-3
    optimizer = None
    lr_scheduler = None

    if parser_args.model == "modernTCN":
        NAME = f"TCN_{parser_args.ls}KS{parser_args.ss}_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from ModernTCN import ModernTCNnew

        model = ModernTCNnew(
            INPUT_CHANNELS,
            TAG_LEN,
            D=D,
            ffn_ratio=parser_args.ratio,
            num_layers=parser_args.num_layers,
            large_sizes=parser_args.ls,
            small_size=parser_args.ss,
            backbone_dropout=0.0,
            head_dropout=parser_args.dp,
            stem=parser_args.learnable_emb,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50 // max(parser_args.batch_size // 50, 1), gamma=0.5
        )

    elif parser_args.model == "Transformer":
        from models.Transformer import Model, Configs

        configs = Configs()
        configs.e_layers = parser_args.num_layers
        configs.dropout = parser_args.dp
        if not parser_args.learnable_emb:
            D = 128 * 5
            configs.d_model = D
            configs.d_ff = D * parser_args.ratio
            configs.n_heads = 4
            NAME = f"TF_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
            model = Model(configs=configs, wide_value_emb=True).to(device)
        else:
            D = 128 * 2
            configs.d_model = D
            configs.d_ff = D * parser_args.ratio
            configs.n_heads = 2
            NAME = f"TF_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
            model = Model(configs=configs, wide_value_emb=False).to(device)

        learning_rate = 0.0
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        lr_lambda = lambda step: (D**-0.5) * min(
            (step + 1) ** -0.5, (step + 1) * 50**-1.5
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif parser_args.model == "iTransformer":
        assert (
            parser_args.learnable_emb == True
        ), "iTransformer 模型必须使用可学习的 emb. TODO"
        D = 128 * 2
        NAME = f"iTransformer_{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.iTransformer import Model, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = configs.d_model * parser_args.ratio
        configs.dropout = parser_args.dp

        model = Model(configs=configs, wide_value_emb=False).to(device)
        learning_rate = 0.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_lambda = lambda step: (D**-0.5) * min(
            (step + 1) ** -0.5, (step + 1) * 50**-1.5
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif parser_args.model == "TimesNet":
        assert (
            parser_args.learnable_emb == True
        ), "TimesNet 模型必须使用可学习的 emb. TODO"
        NAME = f"TimesNet_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.TimesNet import Model, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = D * parser_args.ratio
        configs.dropout = parser_args.dp

        model = Model(configs=configs, wide_value_emb=False).to(device)
        learning_rate = 0.001
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.1
        )

    else:
        raise ValueError("model 选择错误")

    return model, optimizer, lr_scheduler, NAME, learning_rate


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
        from my_datastes import MyDataSet_woEmb as MyDataSet
    else:
        from my_datastes import MyDataSet

    model, optimizer, lr_scheduler, NAME, learn_rate = build_model(
        parser_args, device, NAME
    )

    training_loader, validing_loader, testing_loader, testing_loader_mini = (
        build_dataloaders(parser_args, to_dict_params, F_MASK, MyDataSet)
    )

    loss_record = fit(
        model,
        optimizer,
        lr_scheduler,
        training_loader,
        validing_loader,
        testing_loader_mini,
        parser_args,
        device,
        NAME,
    )

    print(f"learn_rate={learn_rate} batch_size={parser_args.batch_size}")
    if loss_record["train"]:
        print(
            "训练集损失：",
            loss_record["train"][-1],
            "验证集损失：",
            loss_record["vaild"][-1],
            "测试集损失：",
            loss_record["test"][-1],
        )
    print(
        "训练集样本数：",
        len(training_loader.dataset),
        "验证集样本数：",
        len(validing_loader.dataset),
    )
    print(
        "训练集批次数：", len(training_loader), "验证集批次数：", len(validing_loader)
    )
    print("模型信息：", model)
    print("优化器信息：", optimizer)

    loaders_dict = {
        "valid": validing_loader,
        "test_mini": testing_loader_mini,
        "test_full": testing_loader,
    }
    test_final(model, loaders_dict, device, NAME)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
