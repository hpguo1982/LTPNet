from registers import AIITModel
import random
import torchvision.transforms.functional as TF

class AIITRandomVerticalFlipPair(AIITModel):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data["image"] = TF.vflip(data["image"])
            data["label"] = TF.vflip(data["label"])
        return data