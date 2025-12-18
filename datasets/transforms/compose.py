from torchvision.transforms import Compose
from registers import AIITModel


class AIITCompose(Compose, AIITModel):
    """
    Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

        Example:
            >>> transforms.Compose([
            >>>     transforms.CenterCrop(10),
            >>>     transforms.PILToTensor(),
            >>>     transforms.ConvertImageDtype(torch.float),
            >>> ])
    """
    def __init__(self, transforms):
        super().__init__(transforms)


    def __repr__(self):
        string = super().__repr__()
        return string