import torch
from registers import AIITModel

class AIITMinMaxScaling(AIITModel):
    def __init__(self, out_range=(0.0, 1.0)):
        self.out_min, self.out_max = out_range

    def __call__(self, data):

        # tensor shape: (C, H, W)
        image = data["image"]
        x_min = torch.amin(image)
        x_max = torch.amax(image)
        scaled = (image - x_min) / (x_max - x_min)
        data["image"] = scaled * (self.out_max - self.out_min) + self.out_min
        return data
