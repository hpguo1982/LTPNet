import torch

from registers import AIITModel
from torch.nn import Module
from typing import Any, List

class AIITEncoder(Module, AIITModel):

    def __init__(self, backbone, **kwargs: Any):
        super().__init__(**kwargs)
        self.backbone = backbone
        #self.

    def forward(self, x):
        #ret = []

        ret = self.backbone(x)
        return tuple(ret)

    @torch.no_grad()
    def freeze(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = True

    @torch.no_grad()
    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True


