import argparse

import torch
import logging
from engine.single_gpu_trainer import AIITSingleGPUTrainer
from registers import load_config, build_from_config
from utils.logger import AIITLogger
from utils.import_utils import import_symbols
import_symbols("models")
import_symbols("datasets")
import_symbols("evals")
torch.set_float32_matmul_precision("medium")

# ---------- 主程序 ----------
def main(config_path):

    configs = load_config(config_path)
    #configs = load_config("./configs/hcdaa_synapse_config.yaml")
    #configs = load_config("./configs/hcdaa_acdc_config.yaml")
    train_config = configs.get("train_config")
    data_configs = train_config.get("data_configs")
    model_config = train_config.get("model_config")
    schedule_config = train_config.get("schedule_config")
    loss_config = train_config.get("loss_config")
    eval_config = train_config.get("eval_config")

    schedule = load_config(schedule_config)
    logger = build_from_config(schedule["logger"])

    trainer = AIITSingleGPUTrainer(
        data_cfg_path=data_configs,
        model_cfg_path=model_config,
        loss_cfg_path=loss_config,
        eval_cfg_path=eval_config,
        schedule_cfg_path=schedule_config,
        logger=logger
        )
    trainer.train()


def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--config', default="/home/chensen/AIITMedVision-main/configs/ltpnet_isic18.yaml", type=str, help='the config path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args.config)
