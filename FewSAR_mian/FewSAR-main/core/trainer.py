# -*- coding: utf-8 -*-
import datetime
import os
from logging import getLogger
from time import time

import torch
import yaml
from torch import nn
from .data.dataloader import get_dataloader

from queue import Queue
import core.model as arch  # import 模块名 as 别名
from .utils import (
    AverageMeter,
    ModelType,
    SaveType,
    TensorboardWriter,
    count_parameters,
    create_dirs,
    force_symlink,
    get_local_time,
    init_seed,
    init_logger,
    prepare_device,
    save_model,
    get_instance,
)


class Trainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    从config dict构建trainer，设置优化器和模型。
    """

    # 调用下面函数完成初始化过程
    def __init__(self, config):
        self.config = config
        self.device, self.list_ids = self._init_device(config)
        (
            self.result_path,
            self.log_path,
            self.checkpoints_path,
            self.viz_path,
        ) = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.train_meter, self.val_meter, self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.model, self.model_type = self._init_model(config)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self._init_dataloader(config)
        self.optimizer, self.scheduler, self.from_epoch = self._init_optim(config)

    def train_loop(self):
        """
        The normal train loop: train-val-test and save model when val-acc increases.
        正常的训练循环：当 val-acc 增加时训练-val-test 并保存模型。
        """
        best_val_acc = float("-inf")
        best_test_acc = float("-inf")
        experiment_begin = time()
        for epoch_idx in range(self.from_epoch + 1, self.config["epoch"]):
            self.logger.info("============ Train on the train set ============")
            train_acc = self._train(epoch_idx)
            self.logger.info(" * Acc@1 {:.3f} ".format(train_acc))
            self.logger.info("============ Validation on the val set ============")
            val_acc = self._validate(epoch_idx, is_test=False)
            self.logger.info(" * Acc@1 {:.3f} Best acc {:.3f}".format(val_acc, best_val_acc))
            self.logger.info("============ Testing on the test set ============")
            test_acc = self._validate(epoch_idx, is_test=True)
            self.logger.info(" * Acc@1 {:.3f} Best acc {:.3f}".format(test_acc, best_test_acc))
            time_scheduler = self._cal_time_scheduler(experiment_begin, epoch_idx)
            self.logger.info(" * Time: {}".format(time_scheduler))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                self._save_model(epoch_idx, SaveType.BEST)

            if epoch_idx != 0 and epoch_idx % self.config["save_interval"] == 0:
                self._save_model(epoch_idx, SaveType.NORMAL)

            self._save_model(epoch_idx, SaveType.LAST)

            self.scheduler.step()
        self.logger.info(
            "End of experiment, took {}".format(
                str(datetime.timedelta(seconds=int(time() - experiment_begin)))
            )
        )
        self.logger.info("Result DIR: {}".format(self.result_path))


    def _train(self, epoch_idx):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.输出为浮点型的准确率
        """
        #Libfew-shot原有训练
        self.model.train()

        meter = self.train_meter
        meter.reset()
        episode_size = 1 if self.model_type == ModelType.FINETUNING else self.config["episode_size"]

        end = time()

        # print(self.train_loader)
        # exit()
        for batch_idx, batch in enumerate(self.train_loader):
            self.writer.set_step(epoch_idx * len(self.train_loader) + batch_idx * episode_size)

            # visualize the weight 可视化权重
            if self.config["log_paramerter"]:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if "bn" not in name:
                        save_name = name.replace(".", "/")
                        self.writer.add_histogram(save_name, param)

            meter.update("data_time", time() - end)

            # calculate the output 计算输出
            calc_begin = time()
            output, acc, loss = self.model.set_forward_loss(batch)

            # compute gradients 计算梯度
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            meter.update("calc_time", time() - calc_begin)

            # measure accuracy and record loss 测量准确性和记录损失
            meter.update("loss", loss.item())
            meter.update("acc1", acc)

            # measure elapsed time 测量经过的时间
            meter.update("batch_time", time() - end)

            # print the intermediate results 打印中间结果
            if (batch_idx != 0 and (batch_idx + 1) % self.config["log_interval"] == 0) or (
                    batch_idx + 1
            ) * episode_size >= len(self.train_loader):
                info_str = (
                    "Epoch-({}): [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Calc {:.3f} ({:.3f})\t"
                    "Data {:.3f} ({:.3f})\t"
                    "Loss {:.3f} ({:.3f})\t"
                    "Acc@1 {:.3f} ({:.3f})".format(
                        epoch_idx,
                        (batch_idx + 1) * episode_size,
                        len(self.train_loader),
                        meter.last("batch_time"),
                        meter.avg("batch_time"),
                        meter.last("calc_time"),
                        meter.avg("calc_time"),
                        meter.last("data_time"),
                        meter.avg("data_time"),
                        meter.last("loss"),
                        meter.avg("loss"),
                        meter.last("acc1"),
                        meter.avg("acc1"),
                    )
                )
                self.logger.info(info_str)
            end = time()

        return meter.avg("acc1")

    def _validate(self, epoch_idx, is_test=False):
        """
        The val/test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.返回结果为浮点型准确率
        """
        # switch to evaluate mode，转换为验证/测试模式
        self.model.eval()
        self.model.reverse_setting_info()
        meter = self.test_meter if is_test else self.val_meter
        meter.reset()
        episode_size = self.config["episode_size"]

        end = time()
        enable_grad = self.model_type != ModelType.METRIC
        with torch.set_grad_enabled(enable_grad):
            for batch_idx, batch in enumerate(self.test_loader if is_test else self.val_loader):
                self.writer.set_step(
                    int(
                        (epoch_idx * len(self.test_loader) + batch_idx * episode_size)
                        * self.config["tb_scale"]
                    )
                )

                meter.update("data_time", time() - end)

                # calculate the output 计算输出
                calc_begin = time()
                output, acc = self.model.set_forward(batch)
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss 计算准确率和记录损失
                meter.update("acc1", acc)

                # measure elapsed time 测量用时
                meter.update("batch_time", time() - end)

                # 打印结果
                if (batch_idx != 0 and (batch_idx + 1) % self.config["log_interval"] == 0) or (
                        batch_idx + 1
                ) * episode_size >= len(self.val_loader):
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (batch_idx + 1) * episode_size,
                            len(self.val_loader),
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc1"),
                            meter.avg("acc1"),
                        )
                    )
                    self.logger.info(info_str)
                end = time()
        self.model.reverse_setting_info()
        return meter.avg("acc1")

    def _init_files(self, config):
        """
        Init result_path(checkpoints_path, log_path, viz_path) from the config dict.
        从配置字典初始化result_path

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (result_path, log_path, checkpoints_path, viz_path).
        """
        # you should ensure that data_root name contains its true name 需要确保data_root名称包含其真实名称
        # 导入config中的配置文件
        symlink_dir = "{}-{}-{}-{}-{}".format(

            config["data_root"].split("/")[-1],  # 如果字符串最后一位有值，则没有区别，若字符串最后n位都是切割符，split(" ")不会继续切分，split(" ", -1)会继续切分
            config["classifier"]["name"],
            config["backbone"]["name"],
            config["way_num"],
            config["shot_num"],
        )
        result_dir = (
            symlink_dir
            + "{}-{}".format(
                ("-" + config["tag"]) if config["tag"] is not None else "", get_local_time()
            )
            if config["log_name"] is None
            else config["log_name"]
        )
        symlink_path = os.path.join(config["result_root"], symlink_dir)
        result_path = os.path.join(config["result_root"], result_dir)
        # self.logger.log("Result DIR: " + result_path)
        checkpoints_path = os.path.join(result_path, "checkpoints")
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")
        create_dirs([result_path, log_path, checkpoints_path, viz_path])

        force_symlink(result_dir, symlink_path)

        with open(os.path.join(result_path, "config.yaml"), "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(config))

        init_logger(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
        )

        return result_path, log_path, checkpoints_path, viz_path

    def _init_dataloader(self, config):  # 定义初始化dataloader
        """
        Init dataloaders.(train_loader, val_loader and test_loader)

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        # print(self.model_type/)
        train_loader = get_dataloader(config, "train", self.model_type)
        val_loader = get_dataloader(config, "val", self.model_type)
        test_loader = get_dataloader(config, "test", self.model_type)

        return train_loader, val_loader, test_loader

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .
        从配置字典初始化模型（骨干网+分类器）并加载预训练参数或从检查点，然后在必要时并行。

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)  # 导入backbone及参数
        model_kwargs = {  # 导入模型及参数
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)  # 导入分类器及参数

        self.logger.info(model)
        self.logger.info("Trainable params in the model: {}".format(count_parameters(model)))
        # FIXME: May be inaccurate

        if self.config["pretrain_path"] is not None:
            self.logger.info(
                "load pretraining emb_func from {}".format(self.config["pretrain_path"])
            )
            state_dict = torch.load(self.config["pretrain_path"], map_location="cpu")
            msg = model.emb_func.load_state_dict(state_dict, strict=False)

            if len(msg.missing_keys) != 0:
                self.logger.warning("Missing keys:{}".format(msg.missing_keys))
            if len(msg.unexpected_keys) != 0:
                self.logger.warning("Unexpected keys:{}".format(msg.unexpected_keys))

        # 恢复
        if self.config["resume"]:
            resume_path = os.path.join(self.config["resume_path"], "checkpoints", "model_last.pth")
            self.logger.info("load the resume model checkpoints dict from {}.".format(resume_path))
            state_dict = torch.load(resume_path, map_location="cpu")["model"]
            msg = model.load_state_dict(state_dict, strict=False)

            if len(msg.missing_keys) != 0:
                self.logger.warning("missing keys:{}".format(msg.missing_keys))
            if len(msg.unexpected_keys) != 0:
                self.logger.warning("unexpected keys:{}".format(msg.unexpected_keys))

        model = model.to(self.device)  # 如果有超过一块GPU，并行执行
        if len(self.list_ids) > 1:
            parallel_list = self.config["parallel_part"]
            if parallel_list is not None:
                for parallel_part in parallel_list:
                    if hasattr(model, parallel_part):
                        setattr(
                            model,
                            parallel_part,
                            nn.DataParallel(
                                getattr(model, parallel_part),
                                device_ids=self.list_ids,
                            ),
                        )

        return model, model.model_type

    # 初始化优化器
    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.
        从配置中初始化优化器和学习率，如有必要，从检查点加载状态字典。
        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler and epoch_index.
        返回值：
            tuple：优化器、学习率和 epoch_index 的元组。
        """
        params_idx = []
        params_dict_list = []
        # optimizer = Adan(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99))
        if config["optimizer"]["other"] is not None:
            for key, value in config["optimizer"]["other"].items():
                sub_model = getattr(self.model, key)
                params_idx.extend(list(map(id, sub_model.parameters())))
                if value is None:
                    for p in sub_model.parameters():
                        p.requires_grad = False
                else:
                    param_dict = {"params": sub_model.parameters()}
                    if isinstance(value, float):
                        param_dict.update({"lr": value})
                    elif isinstance(value, dict):
                        param_dict.update(value)
                    else:
                        raise Exception("Wrong config in optimizer.other")
                    params_dict_list.append(param_dict)

        params_dict_list.append(
            {"params": filter(lambda p: id(p) not in params_idx, self.model.parameters())}
        )
        optimizer = get_instance(torch.optim, "optimizer", config, params=params_dict_list)
        scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", config, optimizer=optimizer
        )
        self.logger.info(optimizer)
        from_epoch = -1
        if self.config["resume"]:
            resume_path = os.path.join(self.config["resume_path"], "checkpoints", "model_last.pth")
            self.logger.info(
                "load the optimizer, lr_scheduler and epoch checkpoints dict from {}.".format(
                    resume_path
                )
            )
            all_state_dict = torch.load(resume_path, map_location="cpu")
            state_dict = all_state_dict["optimizer"]
            optimizer.load_state_dict(state_dict)
            state_dict = all_state_dict["lr_scheduler"]
            scheduler.load_state_dict(state_dict)
            from_epoch = all_state_dict["epoch"]
            self.logger.info("model resume from the epoch {}".format(from_epoch))

        return optimizer, scheduler, from_epoch

    # 初始化调用GPU
    def _init_device(self, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of deviceand list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(config["device_ids"], config["n_gpu"])
        return device, list_ids

    # 保存模型
    def _save_model(self, epoch, save_type=SaveType.NORMAL):
        """
        Save the model, optimizer, scheduler and epoch.

        TODO

        Args:
            epoch (int): the current epoch index.
            save_type (SaveType, optional): type of (last, best). Defaults to SaveType.NORMAL.
        """
        save_model(
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpoints_path,
            "model",
            epoch,
            save_type,
            len(self.list_ids) > 1,
        )

        if save_type != SaveType.LAST:
            save_list = self.config["save_part"]
            if save_list is not None:
                for save_part in save_list:
                    if hasattr(self.model, save_part):
                        save_model(
                            getattr(self.model, save_part),
                            self.optimizer,
                            self.scheduler,
                            self.checkpoints_path,
                            save_part,
                            epoch,
                            save_type,
                            len(self.list_ids) > 1,
                        )
                    else:
                        self.logger.warning(
                            "{} is not included in {}".format(
                                save_part, self.config["classifier"]["name"]
                            )
                        )

    # 初始化train/val/test阶段的AverageMeter来计算batch_time、data_time、calc_time、loss和acc1的平均值。
    def _init_meter(self):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
            self.writer,
        )
        val_meter = AverageMeter(
            "val",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )
        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )

        return train_meter, val_meter, test_meter

    # 初始化学习率
    def _cal_time_scheduler(self, start_time, epoch_idx):
        """
        Calculate the remaining time and consuming time of the training process.

        Returns:
            str: A string similar to "00:00:00/0 days, 00:00:00". First: comsuming time; Second: total time.
        """
        total_epoch = self.config["epoch"] - self.from_epoch - 1
        now_epoch = epoch_idx - self.from_epoch

        time_consum = datetime.datetime.now() - datetime.datetime.fromtimestamp(start_time)
        time_consum -= datetime.timedelta(microseconds=time_consum.microseconds)
        time_remain = (time_consum * (total_epoch - now_epoch)) / (now_epoch)

        res_str = str(time_consum) + "/" + str(time_remain + time_consum)

        return res_str



