import os
from pathlib import Path
import torch
from tqdm import tqdm

from tools.plot import save_x_y, class2colormap, save_x_y_hat
from utils import AIITLogger
from registers import load_config, build_from_config, AIITModel
import numpy as np


# ---------- 设置随机种子 ----------
class AIITTester(AIITModel):
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
                 data_cfg_path: str = None,
                 model_cfg_path: str = None,
                 testing_cfg_path: str = None,
                 colormap=class2colormap[9],
                 logger: AIITLogger=None):

        self.data_cfg_path = data_cfg_path
        self.model_cfg_path = model_cfg_path
        self.testing_cfg_path = testing_cfg_path
        self.colormap = colormap
        self.logger = logger

        self.logger.info("------初始化------")
        self._set_init_config()

        self.logger.info("------加载测试集------")
        self.test_loader = self._load_data()

        self.logger.info("------加载模型------")
        self.model = self._load_model()

        self.logger.info("------加载评估函数------")
        self.evals = self._load_evaluator()


    def _set_init_config(self):

        #cuda
        testing_cfg = load_config(self.testing_cfg_path)
        testing = testing_cfg.get("testing")

        #cuda
        cuda = testing.get("cuda", None)
        if torch.cuda.is_available() and cuda is not None:
            self._device = torch.device(f"cuda:{cuda}")
        else:
            self._device = torch.device("cpu")

        #checkpoint
        self._checkpoint_path = testing.get("checkpoint")

        #output dir
        self._save_dir = testing.get("save_dir", "./")
        # 如果目录不存在，则创建目录
        os.makedirs(self._save_dir, exist_ok=True)

        self._save_result_file = testing.get("save_result_file")
        self._save_img_dir = testing.get("save_img_dir")
        # 如果目录不存在，则创建目录
        os.makedirs(os.path.join(f"{self._save_dir}", f"{self._save_img_dir}"), exist_ok=True)


    def _load_data(self):
        data_configs = load_config(self.data_cfg_path)
        test_loader = build_from_config(data_configs['test_dataloader'])
        return test_loader

    def _load_model(self):
        model_config = load_config(self.model_cfg_path)
        model = build_from_config(model_config["model"])
        model = self._load_checkpoint(model)
        model = model.to(self._device)
        return model

    def _load_evaluator(self):
        testing_config = load_config(self.testing_cfg_path)
        evals = build_from_config(testing_config["evals"])
        if type(evals) is not list and type(evals) is not tuple:
            evals = [evals]
        return evals

    def _load_checkpoint(self, model):
        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model

    @torch.no_grad()
    def run(self):

        self.model.eval()
        vbar = tqdm(self.test_loader, desc=f"test", unit="batch")

        #清空存储区，以便后面计算
        for eval in self.evals:
            eval.zero_metric()

        for data in vbar:
            image = data['image']
            label= data["label"]
            orig_image = data["orig_image"]
            slice = data["slice"]
            case_name = data["case_name"]

            #支持多slice格式
            if slice[0]:
                image = image.permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                orig_image = orig_image.permute(1, 0, 2, 3)
                orig_image = orig_image.squeeze(dim=1)
                label = label.squeeze(0)
            image = image.to(self._device)
            label = label.to(self._device)
            # output = []
            # for i in range(image.shape[0]):
            #     img = image[i:i+1, :, :, :]
            #     output.append(self.model(img))
            # output = torch.cat(output, dim=0)
            output = self.model(image)

            # =======output figs=====
            if slice[0]:
                out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)

                for depth in range(out.shape[0]):

                    save_x_y(
                        x=(orig_image[depth, :, :]*255).detach().cpu().numpy().astype(np.uint8),
                        y=label[depth, :, :].detach().cpu().numpy().astype(np.uint8),
                        colormap=self.colormap,
                        out=os.path.join(f"{self._save_dir}", f"{self._save_img_dir}", f"{case_name}_{depth}_gt.png")
                    )

                    save_x_y_hat(
                        x=(orig_image[depth, :, :]*255).detach().cpu().numpy().astype(np.uint8),
                        y=label[depth, :, :].detach().cpu().numpy().astype(np.uint8),
                        y_hat=out[depth, :, :].detach().cpu().numpy().astype(np.uint8),
                        colormap=self.colormap,
                        out=os.path.join(f"{self._save_dir}", f"{self._save_img_dir}", f"{case_name}_{depth}_pd.png")
                    )

            for i, eval in enumerate(self.evals):
                eval(output, label)



        des_evals = []
        for eval in self.evals:
            des_evals.append(eval.description)
        self.logger.info("\n"+ "\n".join(des_evals))

        save_dir = os.path.dirname(f"{self._save_dir}/{self._save_result_file}.txt")
        os.makedirs(save_dir, exist_ok=True)

        with open(f"{self._save_dir}/{self._save_result_file}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(des_evals))





