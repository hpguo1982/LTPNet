from models.segmentors.encoders.encoder import AIITEncoder
from typing import Any
from models.segmentors.encoders.libs import FBA
from models.segmentors.backbones import AIITMobileMamba
import torch.nn as nn
from typing import Any, Tuple, List
from torch import Tensor
class AIITLTPEncoder(AIITEncoder):

    def __init__(self, backbone: AIITMobileMamba,
                 img_size=[224, 224],
                 in_dims=[96, 192, 384, 768],
                 **kwargs: Any):

        super().__init__(backbone, **kwargs)

        # 自动根据 backbone 输出通道初始化 FBA 模块
        self.fba_modules: List[FBA] = nn.ModuleList([
            FBA(gate_channels=dim) for dim in in_dims
        ])

    def forward(self, x):
        feats: List[Tensor] = self.backbone(x)  # backbone 输出列表
        # 使用 FBA 增强
        for i, fba in enumerate(self.fba_modules):
            feats[i] = fba(feats[i]) + feats[i]
        return tuple(feats)






