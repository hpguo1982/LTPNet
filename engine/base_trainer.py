import os
import random
from pathlib import Path

import numpy as np
import torch
from utils import AIITLogger
from . import build_optimizer, build_scheduler
from registers import load_config, build_from_config, AIITModel
from torch.cuda.amp import GradScaler

# ---------- 设置随机种子 ----------
class AIITBaseTrainer(AIITModel):
    """
    Class for training model
    :param rank:
        default = 0, the process number
    :param data_cfg_path:
        including the keys: train_dataloader, valid_dataloader, valid_dataloader for dataset configs
    :param model_cfg_path:
        including the key: model for module
    :param schedule_cfg_path:
        including the keys: optimizer and scheduler for optimizer configuration
    :param losses_cfg_path:
        including the key: losses for losses configuration
    :param device:
        cpu or gpu
    """
    def __init__(self,
                 data_cfg_path: str=None,
                 model_cfg_path: str=None,
                 loss_cfg_path: str=None,
                 eval_cfg_path: str=None,
                 schedule_cfg_path: str = None,
                 checkpoint_path: str=None,
                 logger: AIITLogger=None
                 ):
        super().__init__()

        assert os.path.exists(data_cfg_path), f"data path: {data_cfg_path} is not exist!"
        assert os.path.exists(model_cfg_path), f"model config path: {model_cfg_path} is not exist!"
        assert os.path.exists(schedule_cfg_path), f"schedule config path: {schedule_cfg_path} is not exist!"
        assert os.path.exists(loss_cfg_path), f"loss config path: {loss_cfg_path} is not exist!"

        self.data_cfg_path = data_cfg_path
        self.model_cfg_path = model_cfg_path
        self.schedule_cfg_path = schedule_cfg_path
        self.loss_cfg_path = loss_cfg_path
        self.eval_cfg_path = eval_cfg_path
        self.checkpoint_path = checkpoint_path
        self.logger = logger

    # ---------- 训练入口 ----------
    def train(self):
        pass

    # ---------- 私有训练函数 ----------
    def _train_one_epoch(self, epoch):
        """
        :param tbar: 进度条
        :return: (sum of losses, arrary for each loss)
        """
        pass

    @torch.no_grad()
    def _evaluate(self):
        pass

    # ---------- 设置随机种子 ----------
    def _set_seed(self, seed: int = 42, rank: int=0):
        seed = seed + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_data(self):
        pass

    def _load_model(self):
        pass

    def _load_loss(self):
        pass

    def _load_evaluator(self):
        pass

    def _load_optimizer_scheduler(self):
        pass

    def _load_checkpoint(self):
        pass

    def _set_init_config(self):
        pass


