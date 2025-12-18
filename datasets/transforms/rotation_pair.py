from registers import AIITModel
import random
import torchvision.transforms.functional as TF

class AIITRandomRotationPair(AIITModel):
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data["image"] = TF.rotate(data["image"], self.angle)
            data["label"] = TF.rotate(data["label"], self.angle)
        return data