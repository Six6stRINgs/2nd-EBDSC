import torch


def build_model(parser_args, device, NAME):
    D = 128
    learning_rate = 4e-3
    optimizer = None
    lr_scheduler = None

    model = parser_args.model.lower()

    if model == "moderntcn":
        NAME = f"TCN_{parser_args.ls}KS{parser_args.ss}_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from .ModernTCN import ModernTCN

        model = ModernTCN(
            5,
            12,
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

    elif parser_args.model == "bilstm":
        from .BiLSTM import BiLSTM, Configs

        configs = Configs()
        configs.e_layers = parser_args.num_layers
        configs.dropout = parser_args.dp
        if not parser_args.learnable_emb:
            D = 128 * 5
            configs.d_model = D
            NAME = (
                f"BiLSTM_{D}D{parser_args.num_layers}L{parser_args.dp*10:.0f}dp_{NAME}"
            )
            model = BiLSTM(configs=configs, wide_value_emb=True).to(device)
        else:
            D = 128 * 2
            configs.d_model = D
            NAME = (
                f"BiLSTM_{D}D{parser_args.num_layers}L{parser_args.dp*10:.0f}dp_{NAME}"
            )
            model = BiLSTM(configs=configs, wide_value_emb=False).to(device)

        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )

    elif parser_args.model == "transformer":
        from .Transformer import BiLSTM, Configs

        configs = Configs()
        configs.e_layers = parser_args.num_layers
        configs.dropout = parser_args.dp
        if not parser_args.learnable_emb:
            D = 128 * 5
            configs.d_model = D
            configs.d_ff = D * parser_args.ratio
            configs.n_heads = 4
            NAME = f"TF_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
            model = BiLSTM(configs=configs, wide_value_emb=True).to(device)
        else:
            D = 128 * 2
            configs.d_model = D
            configs.d_ff = D * parser_args.ratio
            configs.n_heads = 2
            NAME = f"TF_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
            model = BiLSTM(configs=configs, wide_value_emb=False).to(device)

        learning_rate = 0.0
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        lr_lambda = lambda step: (D**-0.5) * min(
            (step + 1) ** -0.5, (step + 1) * 50**-1.5
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif parser_args.model == "itransformer":
        assert (
            parser_args.learnable_emb == True
        ), "iTransformer 模型必须使用可学习的 emb. TODO"
        D = 128 * 2
        NAME = f"iTransformer_{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.iTransformer import BiLSTM, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = configs.d_model * parser_args.ratio
        configs.dropout = parser_args.dp

        model = BiLSTM(configs=configs, wide_value_emb=False).to(device)
        learning_rate = 0.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_lambda = lambda step: (D**-0.5) * min(
            (step + 1) ** -0.5, (step + 1) * 50**-1.5
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif parser_args.model == "timesnet":
        assert (
            parser_args.learnable_emb == True
        ), "TimesNet 模型必须使用可学习的 emb. TODO"
        NAME = f"TimesNet_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.TimesNet import BiLSTM, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = D * parser_args.ratio
        configs.dropout = parser_args.dp

        model = BiLSTM(configs=configs, wide_value_emb=False).to(device)
        learning_rate = 0.001
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.1
        )

    else:
        raise ValueError("model 选择错误")

    return model, optimizer, lr_scheduler, NAME, learning_rate
