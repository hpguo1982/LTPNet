import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
#from shapely.tests.common import empty_line_string
from skimage.color.adapt_rgb import each_channel
from sympy.physics.units.definitions.dimension_definitions import information
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from utils import AIITLogger
from . import build_optimizer, build_scheduler, AIITBaseTrainer
from registers import load_config, build_from_config
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------- 设置随机种子 ----------
class AIITTrainerDDP(AIITBaseTrainer):
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
                 rank,
                 world_size,
                 data_cfg_path: str=None,
                 model_cfg_path: str=None,
                 losses_cfg_path: str=None,
                 evals_cfg_path: str=None,
                 schedule_cfg_path: str = None,
                 save_dir: str="./checkpoints",
                 checkpoint_path: str=None,

                 main_rank=0):

        super().__init__(
            data_cfg_path=data_cfg_path,
            model_cfg_path=model_cfg_path,
            loss_cfg_path=losses_cfg_path,
            eval_cfg_path=evals_cfg_path,
            schedule_cfg_path=schedule_cfg_path,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path
        )

        self.rank = rank
        self.world_size = world_size
        self.main_rank = main_rank
        self.device = torch.device("cuda:" + str(self.rank))

        if self.rank == self.main_rank:
            self.logger = SimpleLogger(name="MyLogger", log_dir="logs")

        self._setup_distributed()

        # 1) 加载数据
        self._load_data()

        # self.test_loader  = build_from_config(data_configs['test_dataloader'])
        # 2) 加载模型
        self._load_model()

        # 3) 加载损失函数
        self._load_losses()

        # 4) 加载损失函数
        self._load_evaluator()

        # 5) 优化器 & 调度器 & 训练过程输出
        self._load_optimizer_scheduler()

        # 6）加载checkpoint及相关优化器和调度器参数
        self._load_checkpoint()



    def _setup_distributed(self):
        # 本机设置

        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank
        )



    def cleanup(self):
        dist.destroy_process_group()

    # ---------- 训练入口 ----------
    def train(self):

        interval = 0 #for logging
        for epoch in range(self.start_epoch, self.max_epochs + 1):#self.start_epoch用于支持断电续训


            #训练one epoch
            train_loss = self._train_one_epoch(epoch)

            # 调度
            if self.scheduler is not None:
                self.scheduler.step()

            #评估和保存checkpoint
            interval += 1
            if interval%self.log_interval == 0:

                #评估
                self._evaluate()

                # ------- 保存 checkpoint -------
                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                    "loss": train_loss,
                }
                torch.save(state, f"{self.save_dir}/epoch_{epoch}.pth")


    # ---------- 私有训练函数 ----------
    def _train_one_epoch(self, epoch):
        """
        :param tbar: 进度条
        :return: (sum of losses, arrary for each loss)
        """
        # 1️⃣ 保证每个 epoch 的 shuffle 一致
        self.train_sampler.set_epoch(epoch)

        # 2️⃣ tqdm 只在主进程显示进度条
        if self.rank == self.main_rank:  # 只在主进程显示进度条
            loader = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", unit="batch", leave=True) #self.train_loader
        else:
            loader = self.train_loader

        # 3️⃣ 遍历数据
        each_train_loss = np.zeros(len(self.losses))
        self.model.train()

        for data in loader:
            image = data["image"].to(self.rank, non_blocking=True)
            label = data["label"].to(self.rank, non_blocking=True)

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

            each_train_loss += tmp_losses * image.size(0)
            if self.rank == self.main_rank:
                loader.set_postfix(loss=float(sum_loss))

        each_train_loss = each_train_loss / len(self.train_loader)
        train_loss = np.mean(each_train_loss)

        # 记录每个epoch的训练损失
        if self.rank == self.main_rank:
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
        if self.rank == self.main_rank:
            val_loader = tqdm(self.val_loader, desc=f"eval", unit="batch")
        else:
            val_loader = self.val_loader

        #清空存储区，以便后面计算
        for eval in self.evals:
            eval.zero_metric()
        #dict_met = {}

        for data in val_loader:
            image = data['image'].to(self.rank, non_blocking=True)
            label = data['label'].to(self.rank, non_blocking=True)
            output = self.model(image)
            for i, eval in enumerate(self.evals):
                met_value = eval(output, label)
                #dict_met[eval.metric_name] = f"{met_value:.4f}"
            #vbar.set_postfix(**dict_met)

        values = torch.stack([torch.Tensor(eval.metric_value[0], device=self.rank) for eval in enumerate(self.evals)])
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if self.rank == self.main_rank:
            values /= self.world_size
            des_evals = []
            for i, eval in enumerate(self.evals):
                des_evals.append(eval.metric_name + f"{values[i]:.4f}")
            self.logger.info("\n"+ "\n".join(des_evals))

    def _load_data(self):
        if self.rank == self.main_rank:
            self.logger.info("1.加载训练集和验证集")
        data_configs = load_config(self.data_cfg_path)

        # 训练集
        train_dataset = build_from_config(data_configs['train_dataset'])
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        self.train_loader = DataLoader(train_dataset, batch_size=data_configs["train_config"]["batch_size"], sampler=self.train_sampler)

        # 验证集
        if 'valid_dataset' in data_configs:
            val_dataset = build_from_config(data_configs['valid_dataset'])
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank,
                                               shuffle=False)
            self.val_loader = DataLoader(val_dataset, batch_size=data_configs["valid_config"]["batch_size"],
                                           sampler=val_sampler)
        else:
            self.val_loader = None


    def _load_model(self):
        if self.rank == self.main_rank:
            self.logger.info("2.加载模型")
        model_config = load_config(self.model_cfg_path)
        model = build_from_config(model_config["model"])
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[self.rank])

    def _load_losses(self):
        if self.rank == self.main_rank:
            self.logger.info("3.加载损失函数")
        losses_config = load_config(self.loss_cfg_path)
        self.losses = build_from_config(losses_config["losses"])
        if type(self.losses) is not list and type(self.losses) is not tuple:
            self.losses = [self.losses]

    def _load_evaluator(self):
        if self.rank == self.main_rank:
            self.logger.info("4.加载评估函数")
        if self.eval_cfg_path is not None:
            evals_config = load_config(self.eval_cfg_path)
            self.evals = build_from_config(evals_config["evals"])
            if type(self.evals) is not list and type(self.losses) is not tuple:
                self.evals = [self.evals]
        else:
            self.evals = None

    def _load_optimizer_scheduler(self):
        if self.rank == self.main_rank:
            self.logger.info("5.加载优化器 & 调度器")

        schedule_config = load_config(self.schedule_cfg_path)
        self.optimizer = build_optimizer(self.model, schedule_config["optimizer"])
        self.scheduler = build_scheduler(self.optimizer, schedule_config["scheduler"]) if schedule_config.get(
            "scheduler", None) is not None else None

        # 5.1）训练参数
        if self.rank == self.main_rank:
            self.logger.info("5.1 加载训练参数")
        self.max_epochs = schedule_config["training"].get("max_epochs", 100)
        self.seed = schedule_config["training"].get("seed", 42)
        self._set_seed(self.seed)

        # 5.2）output参数
        if self.rank == self.main_rank:
            self.logger.info("5.2 加载输出参数")
            save_dir = schedule_config["output"].get("save_dir", "./checkpoints")
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.save_dir = path.resolve()
        self.log_interval = schedule_config["output"].get("log_interval", 10)
        if self.use_amp is True:
            self.scaler = GradScaler(device_type='cuda')

    def _load_checkpoint(self):
        self.start_epoch = 1
        if self.checkpoint_path is not None:
            if self.rank == self.main_rank:
                self.logger.info("6. 加载checkpoint及相关优化器和调度器参数")
            map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
            state_dict = torch.load(self.checkpoint_path, map_location=map_location)
            self.model.module.load_state_dict(state_dict)

            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.module.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"] + 1