# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn

from core.utils import ModelType
from .init import init_weights

'''
所有分类方法共同的父类
添加新的方法必须要重写set_forward以及set_forward_loss这两个函数，其他的函数都可以根据所实现方法的需要来调用。
为了新添加的方法能够通过反射机制调用到，需要在对应方法类型的目录下的__init__.py文件中加上一行：from NewMethodFileName import *
'''
class AbstractModel(nn.Module):
    def __init__(self, init_type, model_type=ModelType.ABSTRACT, **kwargs):       #初始化函数，用于初始化一些小样本学习中常用的如way，shot，query这样的参数设置。
        super(AbstractModel, self).__init__()

        self.init_type = init_type
        self.model_type = model_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def set_forward(self, *args, **kwargs):          #用于预测阶段调用，返回分类输出以及准确率。
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):    #用于训练阶段调用，返回分类输出、准确率以及前向损失。
        pass

    def forward(self, x):                #重写pytorch的Module中的forward函数，返回backbone的输出。
        out = self.emb_func(x)
        return out

    def train(self, mode=True):          #重写pytorch的Module中的train函数，用于解除BatchNorm层的参数固定。
        super(AbstractModel, self).train(mode)
        # for methods with distiller
        if hasattr(self, "distill_layer"):
            self.distill_layer.train(False)

    def eval(self):                       #重写pytorch的Module中的eval函数，用于固定BatchNorm层的参数。
        return super(AbstractModel, self).eval()

    def _init_network(self):              #用于初始化所有网络。
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):            #用于生成小样本学习的任务中所使用的target。
        local_targets = (
            torch.arange(self.way_num, dtype=torch.long)
            .view(1, -1, 1)
            .repeat(episode_size, 1, self.shot_num + self.query_num)
            .view(-1)
        )
        return local_targets

    def split_by_episode(self, features, mode):         #将输入按照episode_size,way,shot,query切分好便于后续处理。提供了几种切分方式。
        """
        split features by episode and
        generate local targets + split labels by episode
        """
        episode_size = features.size(0) // (self.way_num * (self.shot_num + self.query_num))
        local_labels = (
            self._generate_local_targets(episode_size)
            .to(self.device)
            .contiguous()
            .view(episode_size, self.way_num, self.shot_num + self.query_num)
        )

        if mode == 1:  # input 2D, return 3D(with episode) E.g.ANIL & R2D2
            features = features.contiguous().view(
                episode_size, self.way_num, self.shot_num + self.query_num, -1
            )
            support_features = (
                features[:, :, : self.shot_num, :]
                .contiguous()
                .view(episode_size, self.way_num * self.shot_num, -1)
            )
            query_features = (
                features[:, :, self.shot_num :, :]
                .contiguous()
                .view(episode_size, self.way_num * self.query_num, -1)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif mode == 2:  # input 4D, return 5D(with episode) E.g.DN4
            b, c, h, w = features.shape
            features = features.contiguous().view(
                episode_size,
                self.way_num,
                self.shot_num + self.query_num,
                c,
                h,
                w,
            )
            support_features = (
                features[:, :, : self.shot_num, :, ...]
                .contiguous()
                .view(episode_size, self.way_num * self.shot_num, c, h, w)
            )
            query_features = (
                features[:, :, self.shot_num :, :, ...]
                .contiguous()
                .view(episode_size, self.way_num * self.query_num, c, h, w)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif mode == 3:  # input 4D, return 4D(w/o episode) E.g.realationnet
            b, c, h, w = features.shape
            features = features.contiguous().view(
                self.way_num, self.shot_num + self.query_num, c, h, w
            )
            support_features = (
                features[:, : self.shot_num, :, ...]
                .contiguous()
                .view(self.way_num * self.shot_num, c, h, w)
            )
            query_features = (
                features[:, self.shot_num :, :, ...]
                .contiguous()
                .view(self.way_num * self.query_num, c, h, w)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                episode_size, self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                episode_size, self.way_num * self.query_num
            )
        elif (
            mode == 4
        ):  # finetuning baseline input 2D, return 2D(w/o episode) E.g.baseline set_forward
            features = features.view(self.way_num, self.shot_num + self.query_num, -1)
            support_features = (
                features[:, : self.shot_num, :].contiguous().view(self.way_num * self.shot_num, -1)
            )
            query_features = (
                features[:, self.shot_num :, :].contiguous().view(self.way_num * self.query_num, -1)
            )
            support_target = local_labels[:, :, : self.shot_num].reshape(
                self.way_num * self.shot_num
            )
            query_target = local_labels[:, :, self.shot_num :].reshape(
                self.way_num * self.query_num
            )
        else:
            raise Exception("mode should in [1,2,3,4], not {}".format(mode))

        return support_features, query_features, support_target, query_target

    def reverse_setting_info(self):          #改变小样本学习的way,shot,query等设置。
        (
            self.way_num,
            self.shot_num,
            self.query_num,
            self.test_way,
            self.test_shot,
            self.test_query,
        ) = (
            self.test_way,
            self.test_shot,
            self.test_query,
            self.way_num,
            self.shot_num,
            self.query_num,
        )
