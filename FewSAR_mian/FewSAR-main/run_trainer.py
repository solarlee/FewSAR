# -*- coding: utf-8 -*-
import sys                                       #sys模块包含了与Python解释器和它的环境有关的函数

import torch.cuda

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer

if __name__ == "__main__":
    config = Config("./config/atl_net.yaml").get_config_dict()
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # exit()
    trainer = Trainer(config)
    trainer.train_loop()
