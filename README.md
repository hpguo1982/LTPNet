# 

## Installation

We recommend the following platforms: 

```
Python 3.9 / Pytorch 2.0.1 / NVIDIA A100 80G / CUDA 12.1.0 / Ubuntu
```

In addition, you need to install the necessary packages using the following instructions:

```bash
pip install -r requirements.txt
```

## Prepare data & Pretrained model

#### Dataset:

- **PH2&ISIC16&ISIC17&ISIC18 Dataset**: Download the [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/)  and move into  `dataset/` folder.

#### ImageNet pretrained model:

You should download the pretrained MobileMamba-B4 model from [MonileMamba](https://drive.google.com/file/d/1MiGLHSldK2JpbQEe-7VlmFsaCdugDnYR/view), and then put it in the `model/pretrain/` folder for initialization.

## Training

Using the following command to train & evaluate our model:

```python
# Train PH2
python single_gpu_4_train.py --config configs/ltpnet_ph2.yaml
# Train ISIC16
python single_gpu_4_train.py --config configs/ltpnet_isic16.yaml
# Train ISIC17
python single_gpu_4_train.py --config configs/ltpnet_isic17.yaml
# Train ISIC18
python single_gpu_4_train.py --config configs/ltpnet_isic18.yaml


# Test PH2
python test.py --config configs/ltpnet_ph2.yaml
# Test ISIC16
python test.py --config configs/ltpnet_isic16.yaml
# Test ISIC17
python test.py --config configs/ltpnet_isic17.yaml
# Test ISIC18
python test.py --config configs/ltpnet_isic18.yaml
```

## Acknowledgements

We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet) for making their valuable code & data publicly available.
