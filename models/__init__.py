import torch


def build_model(parser_args, device, NAME):
    # 基础超参数从 parser_args 获取
    D = parser_args.d_model
    learning_rate = parser_args.lr
    
    # 如果用户没有显式指定 d_model (即使用默认值 128)，则保留原有的自动倍增逻辑以兼顾兼容性
    auto_d = (parser_args.d_model == 128)

    model_type = parser_args.model.lower()

    if model_type == "moderntcn":
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

    elif model_type == "bilstm":
        from .BiLSTM import BiLSTM, Configs

        configs = Configs()
        configs.e_layers = parser_args.num_layers
        configs.dropout = parser_args.dp
        
        if auto_d:
            if not parser_args.learnable_emb:
                D = 128 * 5
            else:
                D = 128 * 2
        
        configs.d_model = D
        NAME = f"BiLSTM_{D}D{parser_args.num_layers}L{parser_args.dp*10:.0f}dp_{NAME}"
        model = BiLSTM(configs=configs, wide_value_emb=not parser_args.learnable_emb).to(device)

    elif model_type == "transformer":
        from .Transformer import BiLSTM, Configs # 注意：原代码这里类名如此

        configs = Configs()
        configs.e_layers = parser_args.num_layers
        configs.dropout = parser_args.dp
        
        if auto_d:
            if not parser_args.learnable_emb:
                D = 128 * 5
                default_n_heads = 4
            else:
                D = 128 * 2
                default_n_heads = 2
        else:
            default_n_heads = 8 # parser 的默认值
        
        configs.d_model = D
        configs.d_ff = parser_args.d_ff if parser_args.d_ff is not None else D * parser_args.ratio
        configs.n_heads = parser_args.n_heads if parser_args.n_heads != 8 else default_n_heads
        
        NAME = f"TF_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        model = BiLSTM(configs=configs, wide_value_emb=not parser_args.learnable_emb).to(device)

    elif model_type == "itransformer":
        assert parser_args.learnable_emb, "iTransformer 模型必须使用可学习的 emb."
        if auto_d:
            D = 128 * 2
        
        NAME = f"iTransformer_{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.iTransformer import BiLSTM, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = parser_args.d_ff if parser_args.d_ff is not None else configs.d_model * parser_args.ratio
        configs.dropout = parser_args.dp
        configs.n_heads = parser_args.n_heads

        model = BiLSTM(configs=configs, wide_value_emb=False).to(device)

    elif model_type == "timesnet":
        assert parser_args.learnable_emb, "TimesNet 模型必须使用可学习的 emb."
        if auto_d:
            D = 128
        NAME = f"TimesNet_{D}D{parser_args.num_layers}L{parser_args.ratio}R{parser_args.dp*10:.0f}dp_{NAME}"
        from models.TimesNet import BiLSTM, Configs

        configs = Configs()
        configs.d_model = D
        configs.e_layers = parser_args.num_layers
        configs.d_ff = parser_args.d_ff if parser_args.d_ff is not None else D * parser_args.ratio
        configs.dropout = parser_args.dp

        model = BiLSTM(configs=configs, wide_value_emb=False).to(device)
    else:
        raise ValueError(f"Model {parser_args.model} selection error")

    # --- 统一优化器和调度器创建 ---
    optimizer_type = parser_args.optimizer.lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_type == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler_type = parser_args.scheduler.lower()
    if scheduler_type == "step":
        step_size = parser_args.step_size
        if model_type == "moderntcn":
             step_size = parser_args.step_size // max(parser_args.batch_size // 50, 1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=parser_args.gamma)
    elif scheduler_type == "lambda":
        lr_lambda = lambda step: (D**-0.5) * min((step + 1) ** -0.5, (step + 1) * parser_args.step_size ** -1.5)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=parser_args.step_size, gamma=parser_args.gamma)

    return model, optimizer, lr_scheduler, NAME, learning_rate
