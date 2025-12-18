from .rgvss import build_tiny_0230s, build_small_0229s, load_pretrained_ckpt
from registers import AIITModel
from torch.nn import Module, Sequential
from typing import Any
import torch


class AIITMamba(Module, AIITModel):
    """
    Mamba for versions of tiny_0230s and build_small_0229s
    """
    def __init__(self, ckpt: str = None,
                 version: str = "tiny_0230s",
                 in_channels: int = 3,
                 **kwargs: Any) -> None:
        super().__init__()
        assert version is not None, 'Mamba supports for versions of tiny_0230s and build_small_0229s'
        if version != "tiny_0230s" and version != "small_0229s":
            version = "tiny_0230s"

        if version == "tiny_0230s":
            self.backbone = build_tiny_0230s(ckpt=ckpt, **kwargs)
        else:
            self.backbone = build_small_0229s(ckpt=ckpt, **kwargs)

        #if pretrained is not None:
        #    self.backbone = load_pretrained_ckpt(self.backbone, pretrained)

        self.out_dims = self.backbone.dims
        self.channel_first = self.backbone.channel_first
        self.in_channels = in_channels

        # Patch embedding + stem
        self.layer0 = Sequential(*self.backbone.patch_embed[:5])
        self.layer1 = Sequential(*self.backbone.patch_embed[5:8])

        self.layers = self.backbone.layers
        self.downsamples = self.backbone.downsamples

    def _to_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.channel_first else x.permute(0, 3, 1, 2)

    def forward(self, x):

        if x.shape[1] == 1:
            x = x.repeat(1, self.in_channels, 1, 1)

        ret = []
        x = self.layer0(x)
        x = self.layer1(x)

        for i, layer in enumerate(self.layers):
            #residual = x
            #key = str(x.shape[1])
            #if key in self.rb_dict:  # 残差增强
            #    x = self.rb_dict[key](x) + residual

            x = layer(x)
            ret.append(self._to_channel_first(x))
            x = self.downsamples[i](x)

        return ret
