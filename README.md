<div align="center">

# Snore Source Separation for Denoising
Using UNet1D to learn how to remove non-snore noises from snoring audio signals

</div>


# Overview
This repository contains all the necessary scripts to **prepare**, **preprocess**, and optionally **train** a UNet1D model for **snore source separation** or **denoising**. The approach involves:

- Splitting and organizing raw data into `Dataset/Raw` (via `prepareDataset.py`).
- Downsampling, normalizing, and augmenting the data for training (via `preprocessDataset.py`).
- Training a UNet1D to remove noise from snoring signals (denoising).

# Methods

## Preprocessing
We used kaggle data as starting point and we applied minimal preprocessing involving:
- normalization
- downsampling to 16kHz

## Architectures
We chose 2 well-known architectures for this tasks:
- UNet1D [1]
- CNNAutoEncoder [2]

No significant changes were applied to the original architectures.

## Training
Later too lazy now

## Inference
Later too

# Results
Our strategy yields the following results

| Model                | Denoising | Binary Classification |
|----------------------|-----------|-----------------------|
| UNet1D [1]           |    100    |     100               |
| CNNAutoEncoder [2]   |    100    |     100               |

# How to run

### **Requirements**

- Download dataset from the <a href='https://www.kaggle.com/datasets/tareqkhanemu/snoring?resource=download'>Snoring Kaggle page</a> 
- Place your dataset where you prefer
- Create a virtual environment through `python3 -m venv venv` then `source venv/bin/activate`
- Download dependencies by `pip install -r requirements.txt`
- Optionally, create a wandb account and change the key `wandb_entity` on `config.json` file accordingly. 

All the results were tested on a single NVIDIA RTX A5000 GPU.

### **Prepare and Preprocess data**

```
python prepareDataset.py
```
```
python preprocessDataset.py
```

# References
[1] 

[2] 
