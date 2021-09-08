import warnings
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

import matplotlib.pyplot as plt
import pandas as pd
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_meandice
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
from scipy.ndimage.interpolation import zoom
from torch_geometric.nn import SAGEConv
from torch.cuda.amp import GradScaler, autocast
from src.utils import one_hot, non_zero_acc, np_dice, get_edges
from src.Graph_GNN import UNet_GNN_pass, UNet_GNN_linear
from src.data import get_dataset
from src.unet_gnn_trainer_pass import Trainer
train_set, val_set = get_dataset()


model = UNet_GNN_pass(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

opts = {
    'lr': 5e-4,
    'epochs': 40,
    'batch_size': 32
}
train = Trainer(model, train_set, val_set, opts, torch.device("cuda:6" if torch.cuda.is_available() else "cpu"))
train.train()
