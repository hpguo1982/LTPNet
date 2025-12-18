from registers import AIITModel
import random
import torchvision.transforms.functional as TF

class AIITRandomHorizontalFlipPair(AIITModel):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):

        if random.random() < self.p:
            data["image"] = TF.hflip(data["image"])
            data["label"] = TF.hflip(data["label"])
        return data