from .mix_data_pos import (
    PosMixDatasetCache,
    configure_data_globals,
    get_data_globals,
    read_dfs,
    split_label,
)
import torch.utils.data as Data


def build_dataloaders(config, to_dict_params, f_mask, dataset_cls):
    df_list, test_df_list = read_dfs()
    test_df_split_list = (
        [split_label(item) for item in test_df_list] if config.experiment.mix_test else None
    )

    print("开始进行PDW数据增强......")

    augmentation_kwargs = {
        "sample_ratio": config.augmentation.sample_ratio,
        "drop_sig_ratio": config.augmentation.drop_sig_ratio,
        "cut_len": config.augmentation.cut_len,
        "time_scale_mean": config.augmentation.time_scale_mean,
        "time_scale_std": config.augmentation.time_scale_std,
        "clip_range": tuple(config.augmentation.time_scale_clip),
        "time_shift_exp_scale": config.augmentation.time_shift_exp_scale,
    }
    feature_kwargs = {
        "noise": config.feature_transform.noise,
        "noise_snr": config.feature_transform.noise_snr,
        "rf_mod": config.feature_transform.rf_mod,
        "pw_mod": config.feature_transform.pw_mod,
        "doa_mod": config.feature_transform.doa_mod,
        "rf_scale_range": tuple(config.feature_transform.rf_scale_range),
        "pw_scale_range": tuple(config.feature_transform.pw_scale_range),
        "rf_offset_std": config.feature_transform.rf_offset_std,
        "pw_offset_std": config.feature_transform.pw_offset_std,
        "global_rf_shift_range": tuple(config.feature_transform.global_rf_shift_range),
        "global_pw_shift_range": tuple(config.feature_transform.global_pw_shift_range),
        "global_doa_shift_range": tuple(config.feature_transform.global_doa_shift_range),
        "toa_scale": config.feature_transform.toa_scale,
        "pri_bias": config.feature_transform.pri_bias,
        "pri_scale": config.feature_transform.pri_scale,
        "pw_multiplier": config.feature_transform.pw_multiplier,
    }
    dataset_kwargs = {**augmentation_kwargs, **feature_kwargs, **to_dict_params}

    d_train_base = PosMixDatasetCache(
        df_list,
        config.data_builder.train_size_1,
        config.data_builder.train_size_2,
        is_sequential=(not config.augmentation.enabled),
        is_test=False,
        if_mix_test=config.experiment.mix_test,
        test_df_split_list=test_df_split_list,
        n_jobs=config.data_builder.mix_n_jobs,
        **dataset_kwargs,
    )

    d_valid_base = PosMixDatasetCache(
        df_list,
        config.data_builder.valid_size_1,
        config.data_builder.valid_size_2,
        is_sequential=(not config.augmentation.enabled),
        is_test=False,
        if_mix_test=config.experiment.mix_test,
        test_df_split_list=test_df_split_list,
        n_jobs=config.data_builder.mix_n_jobs,
        **dataset_kwargs,
    )

    d_test_base = PosMixDatasetCache(
        test_df_list,
        is_sequential=True,
        n_jobs=config.data_builder.mix_n_jobs,
        **dataset_kwargs,
    )

    d_test_mini_base = PosMixDatasetCache(
        test_df_list[2],
        config.data_builder.test_mini_size_1,
        config.data_builder.test_mini_size_2,
        is_sequential=(not config.augmentation.enabled),
        is_test=True,
        n_jobs=config.data_builder.mix_n_jobs,
        **dataset_kwargs,
    )

    training_loader = Data.DataLoader(
        dataset_cls(
            d_train_base,
            hard=config.masking.hard_ratio * 0.01,
            f_mask=f_mask,
            config=config,
        ),
        batch_size=config.train.batch_size,
        shuffle=config.data_builder.shuffle_train,
        num_workers=config.runtime.num_workers,
        pin_memory=config.data_builder.pin_memory,
    )
    validing_loader = Data.DataLoader(
        dataset_cls(d_valid_base, hard=None, config=config),
        batch_size=config.train.batch_size,
        shuffle=config.data_builder.shuffle_eval,
        num_workers=config.runtime.num_workers,
        pin_memory=config.data_builder.pin_memory,
    )

    testing_loader = Data.DataLoader(
        dataset_cls(d_test_base, hard=None, config=config),
        batch_size=config.train.batch_size,
        shuffle=config.data_builder.shuffle_eval,
        num_workers=config.runtime.num_workers,
        pin_memory=config.data_builder.pin_memory,
    )

    testing_loader_mini = Data.DataLoader(
        dataset_cls(d_test_mini_base, hard=None, config=config),
        batch_size=config.train.batch_size,
        shuffle=config.data_builder.shuffle_eval,
        num_workers=config.runtime.num_workers,
        pin_memory=config.data_builder.pin_memory,
    )

    return training_loader, validing_loader, testing_loader, testing_loader_mini
