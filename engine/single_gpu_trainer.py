import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
#from shapely.tests.common import empty_line_string
from skimage.color.adapt_rgb import each_channel
from sympy.physics.units.definitions.dimension_definitions import information
from sympy.plotting.intervalmath import interval
from tqdm import tqdm

from utils import AIITLogger
from . import build_optimizer, build_scheduler, AIITBaseTrainer
from registers import load_config, build_from_config
from torch.cuda.amp import autocast, GradScaler

# ---------- 设置随机种子 ----------
class AIITSingleGPUTrainer(AIITBaseTrainer):
    """
    Class for training model
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
                 logger: AIITLogger=None):
        super().__init__(
            data_cfg_path=data_cfg_path,
            model_cfg_path=model_cfg_path,
            loss_cfg_path=loss_cfg_path,
            eval_cfg_path=eval_cfg_path,
            schedule_cfg_path=schedule_cfg_path,
            checkpoint_path=checkpoint_path,
            logger=logger
            )

        self.logger.info("------加载训练参数和输出参数------")
        self._set_init_config()

        self.logger.info("------加载训练集和验证集------")
        self.train_loader, self.val_loader = self._load_data()

        self.logger.info("------加载模型------")
        self.model = self._load_model()

        self.logger.info("------加载损失函数------")
        self.losses = self._load_loss()

        self.logger.info("------加载评估函数------")
        self.evals = self._load_evaluator()

        self.logger.info("------加载优化器 & 调度器------")
        self.optimizer, self.scheduler = self._load_optimizer_scheduler()

        # 7）加载checkpoint及相关优化器和调度器参数
        self.logger.info("-----加载checkpoint------")
        self._load_checkpoint()


    def _set_init_config(self):

        schedule_config = load_config(self.schedule_cfg_path)

        #----训练参数------
        train_cfg = schedule_config["training"]
        self.max_epochs = train_cfg.get("max_epochs", 100)

        #random seed
        self.seed = train_cfg.get("seed", 42)
        self._set_seed(self.seed)

        #F16, F32混合
        self.use_amp = train_cfg.get("use_amp", False)
        if self.use_amp:
            self.scaler = GradScaler()

        #freaze backbone
        self.freeze_backbone = False
        self.freeze_backbone_epochs = train_cfg.get("freeze_backbone_epochs", -1)

        #freeze encoder
        self.freeze_encoder = False
        self.freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", -1)

        #cuda
        cuda = train_cfg.get("cuda", None)
        if torch.cuda.is_available() and cuda is not None:
            self.device = torch.device(f"cuda:{cuda}")
        else:
            self.device = torch.device("cpu")

        #--------验证及保存参数-------
        out_cfg = schedule_config["output"]
        self.log_interval = out_cfg.get("log_interval", 10)
        self.use_val_for_best_model = out_cfg.get("use_val_for_best_model", True)

        save_dir = out_cfg.get("save_dir", "./checkpoints")
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.save_dir = path.resolve()

        self.save_file = out_cfg.get("save_file", "model")

    # ---------- 训练入口 ----------
    def train(self):

        #interval = 0 #for logging
        best_value = 0.0

        for epoch in range(self.start_epoch, self.max_epochs + 1):#self.start_epoch用于支持断电续训

            self.freeze_module(epoch)

            train_loss = self._train_one_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch%self.log_interval == 0:
                # 评估
                eval_value = self._evaluate()

                # ------- 保存 checkpoint -------
                if self.use_val_for_best_model:
                    if eval_value < best_value:
                        continue

                best_value = eval_value
                state = {
                    "ddp": False,
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                    "loss": train_loss,
                }
                torch.save(state, f"{self.save_dir}/{self.save_file}_epoch_{epoch}.pth")

    # ---------- 私有训练函数 ----------
    def _train_one_epoch(self, epoch):
        """
        :param tbar: 进度条
        :return: (sum of losses, arrary for each loss)
        """

        self.model.train()
        each_train_loss = np.zeros(len(self.losses))

        # 进度条
        train_tbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch")

        for data in train_tbar:
            image = data["image"].to(self.device)
            label = data["label"].to(self.device)

            sum_loss = 0.0
            tmp_losses = np.zeros(len(self.losses))

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(image)
                    for i, loss in enumerate(self.losses):
                        tmp_loss = loss(output, label)
                        sum_loss += tmp_loss
                        tmp_losses[i] += tmp_loss
                self.scaler.scale(sum_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(image)
                for i, loss in enumerate(self.losses):
                    tmp_loss = loss(output, label)
                    sum_loss += tmp_loss
                    tmp_losses[i] += tmp_loss
                sum_loss.backward()
                self.optimizer.step()

            each_train_loss += tmp_losses
            train_tbar.set_postfix(loss=float(sum_loss))

        each_train_loss = each_train_loss / len(self.train_loader)
        train_loss = np.sum(each_train_loss)

        # 记录每个epoch的训练损失
        info_each_epoch = f"Train Loss: {train_loss:.4f}| "
        if len(each_train_loss) > 1:
            for i, loss in enumerate(self.losses):
                info_each_epoch += f"{loss.loss_name}: {each_train_loss[i]:.4f} "
        self.logger.info(info_each_epoch)

        return train_loss

    @torch.no_grad()
    def _evaluate(self):
        if self.val_loader is None or self.evals is None:
            return

        self.model.eval()

        vbar = tqdm(self.val_loader, desc=f"eval", unit="batch")

        #清空存储区，以便后面计算
        for eval in self.evals:
            eval.zero_metric()
        #dict_met = {}
        for data in vbar:
            image = data['image']
            label= data["label"]
            slice = data["slice"]
            #支持多slice格式
            if slice[0]:
                image = image.permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                label = label.squeeze(0)

            image = image.to(self.device)
            label = label.to(self.device)

            # output = []
            # for i in range(image.shape[0]):
            #     img = image[i:i+1, :, :, :]
            #     output.append(self.model(img))
            # output = torch.cat(output, dim=0)
            output = self.model(image)
            for i, eval in enumerate(self.evals):
                eval(output, label)

        des_evals = []
        sum_eval = 0.0
        for eval in self.evals:
            for key, values in eval.metric_value.items():
                sum_eval += values[0]
            des_evals.append(eval.description)
        self.logger.info("\n"+ "\n".join(des_evals))
        return sum_eval

    def _load_data(self):
        data_configs = load_config(self.data_cfg_path)
        train_loader = build_from_config(data_configs['train_dataloader'])
        val_loader = build_from_config(data_configs['valid_dataloader']) if data_configs.get('valid_dataloader', None) is not None else None
        return train_loader, val_loader

    def _load_model(self):
        model_config = load_config(self.model_cfg_path)
        model = build_from_config(model_config["model"])
        model = model.to(self.device)
        return model

    def _load_loss(self):
        losses_config = load_config(self.loss_cfg_path)
        losses = build_from_config(losses_config["losses"])
        if type(losses) is not list and type(losses) is not tuple:
            losses = [losses]
        return losses

    def _load_evaluator(self):

        if self.eval_cfg_path is not None:
            evals_config = load_config(self.eval_cfg_path)
            evals = build_from_config(evals_config["evals"])
            if type(evals) is not list and type(evals) is not tuple:
                evals = [evals]
        else:
            evals = None
        return evals

    def _load_optimizer_scheduler(self):
        schedule_config = load_config(self.schedule_cfg_path)
        optimizer = build_optimizer(self.model, schedule_config["optimizer"])
        scheduler = build_scheduler(optimizer, schedule_config["scheduler"]) if schedule_config.get(
            "scheduler", None) is not None else None
        return optimizer, scheduler

    def _load_checkpoint(self):
        self.start_epoch = 1
        if self.checkpoint_path is not None:

            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            if ckpt["ddp"] is False:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt["model"].module)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"] + 1

    def freeze_module(self, current_epoch):
        if current_epoch <= self.freeze_encoder_epochs and not self.freeze_encoder:
                self.model.freeze_encoder()
                self.freeze_encoder = True
        elif self.freeze_encoder and current_epoch > self.freeze_encoder:
                self.model.unfreeze_encoder()
                self.freeze_encoder = False

        if current_epoch <= self.freeze_backbone_epochs and not self.freeze_backbone:
            self.model.freeze_backbone()
            self.freeze_backbone = True
        elif self.freeze_backbone and current_epoch > self.freeze_backbone:
            self.model.unfreeze_backbone()
            self.freeze_backbone = False




