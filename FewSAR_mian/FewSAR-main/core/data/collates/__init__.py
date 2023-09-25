# -*- coding: utf-8 -*-
from .collate_functions import GeneralCollateFunction, FewShotAugCollateFunction
from .contrib import get_augment_method
from ...utils import ModelType


def get_collate_function(config, trfms, mode, model_type):
    """Set the corresponding `collate_fn` by dict.

    + For finetuning-train, return `GeneralCollateFunction`
    + For finetuning-val, finetuning-test and meta/metric-train/val/test, return `FewShotAugCollateFunction`

    Args:
        config (dict): A LFS setting dict.设置字典
        trfms (list): A torchvision transform list.   torchvision中的变换列表
        mode (str): Model mode in ['train', 'test', 'val']   ['train', 'test', 'val'] 中的模型模式
        model_type (ModelType): An ModelType enum value of model.模型的 ModelType 枚举值。

    Returns:
        [type]: [description]
    """
    assert model_type != ModelType.ABSTRACT    #使用assert判断对错，如果出错，将会报错并打印相应语句
    if mode == "train" and model_type == ModelType.FINETUNING:
        collate_function = GeneralCollateFunction(trfms, config["augment_times"])       #是微调则返回GeneralCollateFunction
    else:
        collate_function = FewShotAugCollateFunction(                   #对于finetuning-val、finetuning-test 和 meta/metric-train/val/test，返回 `FewShotAugCollateFunction`
            trfms,
            config["augment_times"],
            config["augment_times_query"],
            config["way_num"] if mode == "train" else config["test_way"],
            config["shot_num"] if mode == "train" else config["test_shot"],
            config["query_num"] if mode == "train" else config["test_query"],
            config["episode_size"],
        )

    return collate_function
