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
from monai.networks.nets import UNet
from src.Graph_GNN import UNet_GNN_pass, UNet_GNN_linear
from src.utils import one_hot, non_zero_acc, np_dice, get_edges
from src.test_models import test_one_case, test_one_model

edges = get_edges()
base = './'
data_path = [base + 'val/' + x for x in os.listdir(base + 'val/')]
names = [x.split('/')[-1].split('_')[0] for x in data_path]
counts = pd.Series(names).value_counts().to_dict()
templete =  './val/{x}_{y}.h5'

model_unet = UNet(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
model_unet.load_state_dict(torch.load('./model_weights/best_unet_weight.pt'))
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
unet_stat = test_one_model(model_unet, device, True, names, counts)
unet_stat.to_csv('./unet_stat.csv')
del model_unet

model_gnn_pass = UNet_GNN_pass(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
model_gnn_pass.load_state_dict(torch.load('./model_weights/best_sage_conv_weight.pt'))
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
gnn_pass_stat = test_one_model(model_gnn_pass, device, False, names, counts)
gnn_pass_stat.to_csv('gnn_pass_stat.csv')
del model_gnn_pass

model_gnn_linear = UNet_GNN_linear(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
model_gnn_linear.load_state_dict(torch.load('./model_weights/best_sage_conv_linear_weight.pt'))
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
gnn_linear_stat = test_one_model(model_gnn_linear, device, False, names, counts)
gnn_linear_stat.to_csv('gnn_linear_stat.csv')
del model_gnn_linear