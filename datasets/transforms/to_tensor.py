from torchvision.transforms.transforms import ToTensor as Tensor
from registers import AIITModel


class AIIT2Tensor(Tensor, AIITModel):
    """
    封装父类：支持动态解析

    Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """
    def __init__(self):
        super().__init__()

    def __repr__(self):
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, grade={self.grade!r})"