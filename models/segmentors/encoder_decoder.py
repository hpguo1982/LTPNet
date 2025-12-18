import torch
from torch.nn import Module

from registers import AIITModel


class AIITEncoderDecoder(Module, AIITModel):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        x = self.encoder(x)

        #转换为至上而下的金子塔数据
        x = list(x)
        x.reverse()
        x = self.decoder(tuple(x))
        return x

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze()

    def freeze_backbone(self) -> None:
        self.encoder.freeze_backbone()

    @torch.no_grad()
    def unfreeze_backbone(self) -> None:
        self.encoder.unfreeze_backbone()


