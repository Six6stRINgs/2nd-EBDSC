from .mix_data_pos import (
    read_dfs,
    split_label,
    PosMixDatasetCache,
    configure_data_globals,
    get_data_globals,
)
import torch.utils.data as Data


def build_dataloaders(parser_args, to_dict_params, F_MASK, MyDataSet):
    df_list, test_df_list = read_dfs()
    test_df_split_list = (
        [split_label(i) for i in test_df_list] if parser_args.mix_test else None
    )

    print(f"开始进行PDW数据增强......")

    # Train / Valid Base
    d_train_base = PosMixDatasetCache(
        df_list,
        100,
        100,
        is_sequential=(not parser_args.aug),
        is_test=False,
        if_mix_test=parser_args.mix_test,
        test_df_split_list=test_df_split_list,
        **to_dict_params,
    )

    d_valid_base = PosMixDatasetCache(
        df_list,
        20,
        50,
        is_sequential=(not parser_args.aug),
        is_test=False,
        if_mix_test=parser_args.mix_test,
        test_df_split_list=test_df_split_list,
        **to_dict_params,
    )

    d_test_base = PosMixDatasetCache(test_df_list, is_sequential=True, **to_dict_params)

    d_test_mini_base = PosMixDatasetCache(
        test_df_list[2],
        20,
        50,
        is_sequential=(not parser_args.aug),
        is_test=True,
        **to_dict_params,
    )

    training_loader = Data.DataLoader(
        MyDataSet(d_train_base, hard=parser_args.hard * 0.01, f_mask=F_MASK),
        batch_size=parser_args.batch_size,
        shuffle=True,
        num_workers=parser_args.num_workers,
        pin_memory=True,
    )
    validing_loader = Data.DataLoader(
        MyDataSet(d_valid_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
        num_workers=parser_args.num_workers,
        pin_memory=True,
    )

    # Test Loaders
    testing_loader = Data.DataLoader(
        MyDataSet(d_test_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
    )

    testing_loader_mini = Data.DataLoader(
        MyDataSet(d_test_mini_base, hard=None),
        batch_size=parser_args.batch_size,
        shuffle=True,
    )

    return training_loader, validing_loader, testing_loader, testing_loader_mini
