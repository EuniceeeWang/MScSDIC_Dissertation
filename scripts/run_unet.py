import matplotlib.pyplot as plt
import pandas as pd
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_meandice
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as opt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import cv2
import h5py
from scipy.ndimage.interpolation import zoom
import torchvision.transforms as T
import random
from src.utils import one_hot, non_zero_acc, np_dice, get_edges
from src.unet_trainer import Trainer
from src.data import get_dataset

model = UNet(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
train_set, val_set = get_dataset()

opts = {
    'lr': 1e-3,
    'epochs': 40,
    'batch_size': 32
}
train = Trainer(model, train_set, val_set, opts, torch.device("cuda:4" if torch.cuda.is_available() else "cpu"))
train.train()