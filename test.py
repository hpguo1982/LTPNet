import argparse

from engine.tester import AIITTester
from registers import load_config, build_from_config
from tools.plot import class2colormap
from utils.import_utils import import_symbols
import_symbols("models")
import_symbols("datasets")
import_symbols("evals")


# ---------- 主程序 ----------
def main(config_path, num_classes):

    configs = load_config(config_path)
    test_config = configs.get("test_config")


    #----------logger------------------
    logger_config = load_config(test_config["testing_config"])
    logger = build_from_config(logger_config["logger"])

    aiit_tester = AIITTester(
        data_cfg_path=test_config["data_config"],
        model_cfg_path=test_config["model_config"],
        testing_cfg_path=test_config["testing_config"],
        colormap=class2colormap[num_classes],
        logger=logger
        )
    aiit_tester.run()

def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--config', default="/home/chensen/AIITMedVision-main/configs/ltpnet_isic18.yaml", type=str, help='the config path')
    parser.add_argument("--num_classes",default=1, type=int, help='the number of classes for current dataset')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args.config, args.num_classes)