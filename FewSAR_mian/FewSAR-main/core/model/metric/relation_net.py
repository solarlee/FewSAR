# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/cvpr/SungYZXTH18,
  author    = {Flood Sung and
               Yongxin Yang and
               Li Zhang and
               Tao Xiang and
               Philip H. S. Torr and
               Timothy M. Hospedales},
  title     = {Learning to Compare: Relation Network for Few-Shot Learning},
  booktitle = {2018 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2018, Salt Lake City, UT, USA, June 18-22, 2018},
  pages     = {1199--1208},
  year      = {2018},
  url       = {http://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html},
  doi       = {10.1109/CVPR.2018.00131}
}
https://arxiv.org/abs/1711.06025

Adapted from https://github.com/floodsung/LearningToCompare_FSL.
"""
import torch
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class RelationLayer(nn.Module):                                          #用于计算两个拼接在一起的特征的相关性系数,借用nn中的module类来定义函数
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3):
        super(RelationLayer, self).__init__()                            #super（）用于调用父类
        self.layers = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=0),
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),          #momentum：一个用于运行过程中均值和方差的一个估计参数；affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
            nn.ReLU(inplace=True),                                   #非线性激活函数
            nn.MaxPool2d(2),                                     #对于输入信号的输入通道，提供2维最大池化（max pooling）操作；如果padding不是0，会在输入的每一边添加相应数目0
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=0),         #kernel_size:一个整数或2个整数的元组/列表，指定二维卷积窗口的高度和宽度。padding: one of "valid" or "same" (不区分大小写).[注：卷积会导致输出图像越来越小，图像边界信息丢失，若想保持卷积后的图像大小不变，需要设置padding参数为same]
            nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )                                                             #多层卷积网络

    #神经网络中，我们一般输入都是二维的tensor矩阵（batch，input_size），但其实输入的维度是不做限制的。如果是三维的输入，会将前两维的数据先乘一起，然后在做计算，实际上还是单层神经网络的计算。
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * feat_height * feat_width, 8),   #nn.Linear(输入数据的形状,输出数据的形状,bias默认是True)
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x):                  #前向传播，x的四个参数分别为batch大小、特征向量维度，和特征的长宽
        #print(x.shape)
        out = self.layers(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out


class RelationNet(MetricModel):                                     #定义RN类，使用度量学习模型
    def __init__(self, feat_dim=64, feat_height=3, feat_width=3, **kwargs):       #在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self。其作用相当于java中的this，表示当前类的对象，可以调用当前类中的属性和方法。
        super(RelationNet, self).__init__(**kwargs)                #kwargs就是当你传入key=value是存储的字典
        self.feat_dim = feat_dim
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.relation_layer = RelationLayer(self.feat_dim, self.feat_height, self.feat_width)
        self.loss_func = nn.CrossEntropyLoss()

    #设置前向传播，用于预测阶段，返回分类输出和准确率
    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch         #在batch中将样本与标签分开
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)

        relation_pair = self._calc_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pair).view(-1, self.way_num)

        acc = accuracy(output, query_target.view(-1))
        return output, acc

    #设置向前传递损失，用于训练阶段调用，返回分类输出、准确率以及前项损失
    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)

        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, mode=2)

        relation_pair = self._calc_pairs(query_feat, support_feat)
        output = self.relation_layer(relation_pair).view(-1, self.way_num)

        loss = self.loss_func(output, query_target.view(-1))
        acc = accuracy(output, query_target.view(-1))
        return output, acc, loss

    #通过拼接，计算特征相似度
    def _calc_pairs(self, query_feat, support_feat):
        """

        :param query_feat: (task_num, query_num * way_num, feat_dim, feat_width, feat_height)
        :param support_feat: (task_num, support_num * way_num, feat_dim, feat_width, feat_height)
        :return: query_num * way_num * way_num, feat_dim, feat_width, feat_height
        """
        t, _, c, h, w = query_feat.size()
        # t, w, wq, c, h, w -> t, wq, w, c, h, w
        query_feat = query_feat.unsqueeze(1).repeat(1, self.way_num, 1, 1, 1, 1)
        query_feat = torch.transpose(query_feat, 1, 2)

        # t, w, s, c, h, w -> t, 1, w, c, h, w -> t, wq, w, c, h, w
        support_feat = support_feat.view(t, self.way_num, self.shot_num, c, h, w)
        support_feat = (
            torch.sum(support_feat, dim=(2,))
            .unsqueeze(1)
            .repeat(1, self.way_num * self.query_num, 1, 1, 1, 1)
        )

        # t, wq, w, 2c, h, w -> twqw, 2c, h, w
        relation_pair = torch.cat((query_feat, support_feat), dim=3).view(-1, c * 2, h, w)
        return relation_pair
