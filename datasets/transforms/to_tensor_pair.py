import torch
from torchvision.transforms.functional import to_tensor

from registers import AIITModel


class AIIT2TensorPair(AIITModel):
    def __init__(self):
        super().__init__()

    def __call__(self, data):

        data["image"] = torch.tensor(data["image"]).permute(2, 0, 1).float()
        data["label"] = torch.tensor(data["label"]).permute(2, 0, 1)
        data["orig_image"] = to_tensor(data["orig_image"]).permute(2, 0, 1)
        return data