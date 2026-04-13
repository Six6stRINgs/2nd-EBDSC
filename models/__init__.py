import torch


def build_model(config, device, name):
    model_cfg = config.model
    experiment_cfg = config.experiment
    optimizer_cfg = config.optimizer
    scheduler_cfg = config.scheduler
    task_cfg = config.task
    wide_cfg = config.wide_value_embedding

    d_model = model_cfg.d_model
    learning_rate = optimizer_cfg.lr
    auto_d = model_cfg.auto_scale_d_model
    model_type = experiment_cfg.model_name.lower()

    wide_value_dim = wide_cfg.pos_d * task_cfg.num_features
    learnable_emb_dim = wide_cfg.pos_d * 2

    if model_type == "moderntcn":
        name = (
            f"TCN_{model_cfg.large_kernel_size}KS{model_cfg.small_kernel_size}_"
            f"{d_model}D{model_cfg.num_layers}L{model_cfg.ratio}R{model_cfg.dropout*10:.0f}dp_{name}"
        )
        from .ModernTCN import ModernTCN

        model = ModernTCN(
            task_cfg.num_features,
            task_cfg.num_classes,
            D=d_model,
            ffn_ratio=model_cfg.ratio,
            num_layers=model_cfg.num_layers,
            large_sizes=model_cfg.large_kernel_size,
            small_size=model_cfg.small_kernel_size,
            backbone_dropout=0.0,
            head_dropout=model_cfg.dropout,
            stem=experiment_cfg.learnable_emb,
        ).to(device)

    elif model_type == "bilstm":
        from .BiLSTM import BiLSTM, Configs

        configs = Configs()
        configs.e_layers = model_cfg.num_layers
        configs.dropout = model_cfg.dropout
        configs.num_classes = task_cfg.num_classes
        configs.dim = model_cfg.bilstm_hidden_proj_dim

        if auto_d:
            d_model = (
                wide_value_dim
                if not experiment_cfg.learnable_emb
                else learnable_emb_dim
            )

        configs.d_model = d_model
        name = f"BiLSTM_{d_model}D{model_cfg.num_layers}L{model_cfg.dropout*10:.0f}dp_{name}"
        model = BiLSTM(
            configs=configs, wide_value_emb=not experiment_cfg.learnable_emb
        ).to(device)

    elif model_type == "transformer":
        from .Transformer import Transformer, Configs

        configs = Configs()
        configs.e_layers = model_cfg.num_layers
        configs.dropout = model_cfg.dropout
        configs.num_class = task_cfg.num_classes
        configs.enc_in = task_cfg.num_features
        configs.seq_len = task_cfg.window_size

        if auto_d:
            if not experiment_cfg.learnable_emb:
                d_model = wide_value_dim
                default_n_heads = 4
            else:
                d_model = learnable_emb_dim
                default_n_heads = 2
        else:
            default_n_heads = model_cfg.n_heads

        configs.d_model = d_model
        configs.d_ff = (
            model_cfg.d_ff if model_cfg.d_ff is not None else d_model * model_cfg.ratio
        )
        configs.n_heads = model_cfg.n_heads
        if auto_d and model_cfg.n_heads == 8:
            configs.n_heads = default_n_heads

        name = (
            f"TF_{d_model}D{model_cfg.num_layers}L{model_cfg.ratio}R"
            f"{model_cfg.dropout*10:.0f}dp_{name}"
        )
        model = Transformer(
            configs=configs, wide_value_emb=not experiment_cfg.learnable_emb
        ).to(device)

    elif model_type == "itransformer":
        assert experiment_cfg.learnable_emb, "iTransformer 模型必须使用可学习的 emb."
        if auto_d:
            d_model = learnable_emb_dim

        name = (
            f"iTransformer_{d_model}D{model_cfg.num_layers}L{model_cfg.ratio}R"
            f"{model_cfg.dropout*10:.0f}dp_{name}"
        )
        from models.iTransformer import iTransformer, Configs

        configs = Configs()
        configs.d_model = d_model
        configs.e_layers = model_cfg.num_layers
        configs.seq_len = task_cfg.window_size
        configs.enc_in = task_cfg.num_features
        configs.d_ff = (
            model_cfg.d_ff if model_cfg.d_ff is not None else d_model * model_cfg.ratio
        )
        configs.dropout = model_cfg.dropout
        configs.n_heads = model_cfg.n_heads
        configs.num_class = task_cfg.num_classes

        model = iTransformer(configs=configs, wide_value_emb=False).to(device)

    elif model_type == "timesnet":
        assert experiment_cfg.learnable_emb, "TimesNet 模型必须使用可学习的 emb."
        if auto_d:
            d_model = wide_cfg.pos_d
        name = (
            f"TimesNet_{d_model}D{model_cfg.num_layers}L{model_cfg.ratio}R"
            f"{model_cfg.dropout*10:.0f}dp_{name}"
        )
        from models.TimesNet import TimesNet, Configs

        configs = Configs()
        configs.d_model = d_model
        configs.e_layers = model_cfg.num_layers
        configs.seq_len = task_cfg.window_size
        configs.enc_in = task_cfg.num_features
        configs.d_ff = (
            model_cfg.d_ff if model_cfg.d_ff is not None else d_model * model_cfg.ratio
        )
        configs.dropout = model_cfg.dropout
        configs.num_class = task_cfg.num_classes

        model = TimesNet(configs=configs, wide_value_emb=False).to(device)
    else:
        raise ValueError(f"Model {experiment_cfg.model_name} selection error")

    optimizer_type = optimizer_cfg.name.lower()
    optimizer_kwargs = {"lr": learning_rate, "weight_decay": optimizer_cfg.weight_decay}
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler_type = scheduler_cfg.name.lower()
    if scheduler_type == "step":
        step_size = scheduler_cfg.step_size
        if model_type == "moderntcn":
            step_size = scheduler_cfg.step_size // max(config.train.batch_size // 50, 1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(step_size, 1), gamma=scheduler_cfg.gamma
        )
    elif scheduler_type == "lambda":
        warmup_steps = scheduler_cfg.lambda_warmup_steps

        def lr_lambda(step):
            return (d_model**-0.5) * min(
                (step + 1) ** -0.5,
                (step + 1) * warmup_steps**-1.5,
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(scheduler_cfg.step_size, 1),
            gamma=scheduler_cfg.gamma,
        )

    return model, optimizer, lr_scheduler, name, learning_rate
