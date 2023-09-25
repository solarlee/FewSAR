# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms

from core.data.dataset import GeneralDataset
from .collates import get_collate_function, get_augment_method
from .samplers import CategoriesSampler
from ..utils import ModelType

# ToTensor & Norm部分使用同一组均值和方差，你可以根据数据集特性重新设置该值
MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]


def get_dataloader(config, mode, model_type):
    """Get the dataloader corresponding to the model type and training phase.
    获取模型类型和训俩阶段的dataloader。

    According to the config dict, the training phase and model category, select the appropriate transforms, set the corresponding sampler and collate_fn, and return the corresponding dataloader.
    根据config dict，训练阶段和模型类别，选择合适的transforms，设置对应的sampler和collate_fn，返回对应的dataloader。

    Args:
        config (dict): A LibFewShot setting dict
        mode (str): mode in train/test/val
        model_type (ModelType): model type in meta/metric//finetuning

    Returns:
        Dataloader: The corresponding dataloader.
    """
    assert model_type != ModelType.ABSTRACT

    trfms_list = []
    # print(config["image_size"])
    # exit()
    # Add user's trfms here (or in get_augment_method())
    if mode == "train" and config["augment"]:                   #训练时处理图像尺寸并进行数据增广
        if config["image_size"] == 224:
            trfms_list.append(transforms.Resize((256, 256)))     #调整图像尺寸Resize()
            trfms_list.append(transforms.RandomCrop((224, 224)))      #随机裁剪RandomCrop()
        elif config["image_size"] == 84:
            trfms_list.append(transforms.Resize((96, 96)))
            trfms_list.append(transforms.RandomCrop((84, 84)))
        # for MTL -> alternative solution: use avgpool(ks=11)
        elif config["image_size"] == 80:
            # MTL use another MEAN and STD
            trfms_list.append(transforms.Resize((92, 92)))
            trfms_list.append(transforms.RandomResizedCrop(88))
            trfms_list.append(transforms.CenterCrop((80, 80)))
            trfms_list.append(transforms.RandomHorizontalFlip())
        else:
            raise RuntimeError

        aug_method = get_augment_method(config)
        trfms_list += aug_method
    else:                                        #验证和测试时直接处理图像尺寸，不做数据增广
        if config["image_size"] == 224:
            trfms_list.append(transforms.Resize((256, 256)))
            trfms_list.append(transforms.CenterCrop((224, 224)))
        elif config["image_size"] == 84:
            trfms_list.append(transforms.Resize((96, 96)))
            trfms_list.append(transforms.CenterCrop((84, 84)))
        # for MTL -> alternative solution: use avgpool(ks=11)
        elif config["image_size"] == 80:
            trfms_list.append(transforms.Resize((92, 92)))
            trfms_list.append(transforms.CenterCrop((80, 80)))
        else:
            raise RuntimeError

    trfms_list.append(transforms.ToTensor())
    trfms_list.append(transforms.Normalize(mean=MEAN, std=STD))         #使用transforms.Normalize(mean, std)对图像按通道进行标准化，即减去均值，再除以方差。这样做可以加快模型的收敛速度，mean和std分别为均值和方差序列
    trfms = transforms.Compose(trfms_list)              #transforms.Compose()类用来组合多个torchvision.transforms操作。参数为transform对象

    dataset = GeneralDataset(           #导入数据集，并调用上面的mode调整图像尺寸
        data_root=config["data_root"],
        mode=mode,
        use_memory=config["use_memory"],
    )

    collate_function = get_collate_function(config, trfms, mode, model_type)    #整理数据，调用init.py


    '''
    基于微调的模型使用了语言模型而非深度神经网络
    该模型分为3 个阶段:(1) 语言模型预训练;(2) 语言模型微调;(3) 分类器微调.该模型的创新点在于改变学习速率来微调语言模型
    '''
    if mode == "train" and model_type == ModelType.FINETUNING:
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_function,
        )
    else:
        sampler = CategoriesSampler(
            label_list=dataset.label_list,
            label_num=dataset.label_num,
            episode_size=config["episode_size"],
            episode_num=config["train_episode"] if mode == "train" else config["test_episode"],
            way_num=config["way_num"] if mode == "train" else config["test_way"],
            image_num=config["shot_num"] + config["query_num"]
            if mode == "train"
            else config["test_shot"] + config["test_query"],
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_function,
        )

    return dataloader
