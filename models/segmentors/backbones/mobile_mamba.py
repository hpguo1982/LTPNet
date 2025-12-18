from torch.nn import Module
from registers import AIITModel
from models.segmentors.backbones.mobilemamba.mobilemamba.mobilemamba import Mobilemamba
class AIITMobileMamba(Module, AIITModel):
    def __init__(self, ckpt: str = None):
        super().__init__()
        self.mobilemambabackbone = Mobilemamba(ckpt)
    def forward(self,x):
        # feats = self.mobilemambabackbone(x)
        return self.mobilemambabackbone(x)
# import torch
# model = MobileMambaBackbone().to("cuda")
# print(model(torch.randn(2,3,640,640).to("cuda")))