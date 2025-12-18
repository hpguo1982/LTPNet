import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from . import build_optimizer, build_scheduler, AIITBaseTrainer
from registers import load_config, build_from_config


# ====== 简单数据集 ======
class RandomDataset(Dataset):
    def __init__(self, size=1000, length=10):
        self.len = size
        self.data = torch.randn(size, length)

    def __getitem__(self, index):
        return self.data[index], torch.tensor(1.0)  # 伪标签

    def __len__(self):
        return self.len


# ====== 模型 ======
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3))

    def forward(self, x):
        return self.conv(x)


# ====== 训练器类 ======
class DDPTrainerTest(AIITBaseTrainer):

    def __init__(self, rank, world_size, data_cfg_path, model_cfg_path, loss_cfg_path, eval_cfg_path, schedule_cfg_path, epochs=5, batch_size=32, lr=0.01):
        super().__init__(
            data_cfg_path=data_cfg_path,
            model_cfg_path=model_cfg_path,
            loss_cfg_path=loss_cfg_path,
            eval_cfg_path=eval_cfg_path,
            schedule_cfg_path=schedule_cfg_path,
            #save_dir=save_dir,
            #checkpoint_path=checkpoint_path,
            #use_amp=use_amp
        )

        self.rank = rank
        self.world_size = world_size

        self.data_cfg_path = data_cfg_path
        self.model_cfg_path = model_cfg_path
        self.loss_cfg_path = loss_cfg_path
        self.eval_cfg_path = eval_cfg_path
        self.schedule_cfg_path = schedule_cfg_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr


        self.device = self.setup_environ()

        # 1) 加载数据
        self.train_loader, self.train_sampler, self.val_loader = self._load_data()

        # 2) 加载模型
        self.ddp_model = self._load_model()

        # 3) 加载损失函数
        self.losses = self._load_loss()

        # 4) 加载损失函数
        self.evals = self._load_evaluator()

        # 5) 优化器 & 调度器
        self.optimizer, self.scheduler = self._load_optimizer_scheduler()

        #6)配置其它训练参数和输出参数
        self._set_init_config()

    def setup_environ(self):
        """初始化分布式环境"""
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank
        )
        torch.manual_seed(0)
        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)
        return device

    def _load_data(self):
        data_configs = load_config(self.data_cfg_path)
        # 训练集
        dataset = build_from_config(data_configs['train_dataset'])
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        batch_size = data_configs["train_config"]["batch_size"]
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        # 验证集
        if 'valid_dataset' in data_configs:
            val_dataset = build_from_config(data_configs['valid_dataset'])
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank,
                                             shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=data_configs["valid_config"]["batch_size"],
                                         sampler=val_sampler)
        else:
            val_loader = None

        return train_loader, sampler, val_loader

    def _load_model(self):
        # if self.rank == self.main_rank:
        #     self.logger.info("2.加载模型")
        model_config = load_config(self.model_cfg_path)
        model = build_from_config(model_config["model"])
        model = model.to(self.device)  # SimpleModel().to(device)
        ddp_model = DDP(model, device_ids=[self.rank])
        return ddp_model

    def _load_loss(self):
        losses_config = load_config(self.loss_cfg_path)
        losses = build_from_config(losses_config["losses"])
        if type(losses) is not list and type(losses) is not tuple:
            losses = [losses]
        return losses

    def _load_evaluator(self):
        #self.logger.info("4.加载评估函数")
        if self.eval_cfg_path is not None:
            evals_config = load_config(self.eval_cfg_path)
            evals = build_from_config(evals_config["evals"])
            if type(evals) is not list and type(evals) is not tuple:
                evals = [evals]
        else:
            evals = None
        return evals

    def _load_optimizer_scheduler(self):
        #if self.rank == self.main_rank:
        #    self.logger.info("5.加载优化器 & 调度器")

        schedule_config = load_config(self.schedule_cfg_path)
        optimizer = build_optimizer(self.ddp_model, schedule_config["optimizer"])
        scheduler = build_scheduler(optimizer, schedule_config["scheduler"]) if schedule_config.get(
            "scheduler", None) is not None else None
        return optimizer, scheduler

    def _set_init_config(self):

        schedule_config = load_config(self.schedule_cfg_path)

        #----训练参数------
        train_cfg = schedule_config["training"]
        self.max_epochs = train_cfg.get("max_epochs", 100)

        #random seed
        self.seed = train_cfg.get("seed", 42)
        self._set_seed(self.seed + self.rank)

        #F16, F32混合
        self.use_amp = train_cfg.get("use_amp", False)
        if self.use_amp:
            self.scaler = GradScaler()

        #--------验证及保存参数-------
        self.log_interval = schedule_config["output"].get("log_interval", 10)
        self.save_dir = schedule_config["output"].get("save_dir", "./checkpoints")

    def cleanup(self):
        """销毁分布式环境"""
        dist.destroy_process_group()

    def train_worker(self):
        """单个进程的训练逻辑"""
        print(f"[Rank {self.rank}] starting training...")

        # 训练循环
        for epoch in range(self.epochs):

            self.train_sampler.set_epoch(epoch)
            epoch_loss = self._train_one_epoch(epoch)

            print(f"[Rank {self.rank}] Epoch {epoch}, Loss: {epoch_loss/len(self.train_loader):.4f}")

        self.cleanup()

    def _train_one_epoch(self, epoch):

        self.ddp_model.train()

        epoch_loss = 0.0
        # 进度条
        if self.rank == 0:
            loader = tqdm(self.train_loader, desc=f"Epoch {epoch}", unit="batch")
        else:
            loader = self.train_loader

        for batch, data in enumerate(loader):
            image = data["image"].to(self.device)
            label = data["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    output = self.ddp_model(image)
                    for i, loss in enumerate(self.losses):
                        tmp_loss = loss(output, label)
                #         sum_loss += tmp_loss
                #         tmp_losses[i] += tmp_loss
                #
                # self.scaler.scale(sum_loss).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
            else:
                output = self.ddp_model(image)
                for i, loss in enumerate(self.losses):
                    tmp_loss = loss(output, label)
                #     sum_loss += tmp_loss
                #     tmp_losses[i] += tmp_loss
                # sum_loss.backward()
                # self.optimizer.step()

            #self.optimizer.zero_grad()
            #output = self.ddp_model(image)
            # loss = criterion(output, target.unsqueeze(1))
            # loss.backward()
            # optimizer.step()
            #print(f"[Rank {self.rank}] Epoch {epoch}, Loss: {epoch_loss / len(self.train_loader):.4f}")

            # epoch_loss += loss.item()
        return epoch_loss





