import torch
from torch.utils.data import Dataset
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image
from registers import AIITModel


class AIITISICDataset(Dataset, AIITModel):
    def __init__(self, base_dir, transformer, train=True):
        super().__init__()
        if train:
            image_paths = base_dir + 'train/images/'
            mask_paths = base_dir + 'train/masks/'
        else:
            image_paths = base_dir + 'val/images/'
            mask_paths = base_dir + 'val/masks/'

        images_list = sorted(os.listdir(image_paths))
        masks_list = sorted(os.listdir(mask_paths))
        self.data = []
        for i in range(len(images_list)):
            img_path = image_paths + images_list[i]
            mask_path = mask_paths + masks_list[i]
            self.data.append([img_path, mask_path])

        self.transformer = transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        sample = {'image': img, 'label': msk, "orig_image": img, "case_name": file_name}
        sample = self.transformer(sample)
        if sample['label'].shape[0] == 1:
            sample['label'] = torch.squeeze(sample["label"], axis=0)

        sample["slice"] = False

        return sample

    def __len__(self):
        return len(self.data)

