
from registers import AIITModel
import torchvision.transforms.functional as TF


class AIITNormalizePair(AIITModel):
    def __init__(self, mean=157.561, std=26.706):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data["image"] = TF.normalize(data["image"], mean=self.mean, std=self.std)
        #img_normalized = ((img - torch.min(img))
        #                  / (torch.max(img) - torch.min(img))) * 255.
        return data
        #img_tensor = transforms.ToTensor()(img)  # 自动缩放到[0,1]范围
        # mask_tensor = transforms.ToTensor()(msk).long()  # 保持整数标签
        # return img_tensor, mask_tensor