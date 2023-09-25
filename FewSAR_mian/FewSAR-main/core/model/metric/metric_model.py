# -*- coding: utf-8 -*-
from abc import abstractmethod

from core.model.abstract_model import AbstractModel
from core.utils import ModelType

#所有度量学习的父类，均按照以下方式定义
class MetricModel(AbstractModel):
    def __init__(self, init_type="normal", **kwargs):
        super(MetricModel, self).__init__(init_type, ModelType.METRIC, **kwargs)

    @abstractmethod
    def set_forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_forward_loss(self, *args, **kwargs):
        pass

    def forward(self, x):
        out = self.emb_func()
        return out
