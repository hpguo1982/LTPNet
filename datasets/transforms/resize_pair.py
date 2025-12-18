from registers import AIITModel
import torchvision.transforms.functional as TF


class AIITResizePair(AIITModel):
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        data["image"] = TF.resize(data["image"], [self.size_h, self.size_w])
        data["label"] = TF.resize(data["label"], [self.size_h, self.size_w])
        data["orig_image"] = TF.resize(data["orig_image"], [self.size_h, self.size_w])
        return data
