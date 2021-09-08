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

templete =  './val/{x}_{y}.h5' # string templete
def test_one_case(name, counts, model, device, unet=False):
    count = counts[name]
    real = [] # for stroing the ground truth
    out = [] # for storing the output
    for i in range(1, count-2):
        paths = []
        # get all the paths
        for j in range(i-1, i+2):
            paths.append(templete.format(x=name, y=j))
            
        # this part is consistant with the model's triaining loader
        xs = []
        f = h5py.File(paths[1], 'r')
        y = f['gt'][:]
        real.append(y)
        f.close()
        for path in paths:
            f = h5py.File(path, 'r')
            x = f['ct'][:].astype(np.float)
            x = torch.from_numpy(x)
            x = x.float().view(1, 400, 400)
            xs.append(x)  
            f.close()
        x = torch.cat(xs, dim=0).view(1, 3, 400, 400)
        x = x.to(device)
        
        # since the gnn models need the edges information, we need to seperate them
        if unet:
            outputs = model(x)
        else:
            outputs = model(x, device, edges)
        outputs = torch.nn.functional.softmax(outputs, 1) # outputs should be pass into the softmax layer
        _, predicted = torch.max(outputs.data, 1) # get the max value as the final prediction
        predicted = predicted.view(400, 400).detach().cpu().numpy()
        out.append(predicted)
        
    out = np.array(out)
    real = np.array(real)
    return np.mean(out == real), non_zero_acc(out, real), np_dice(out, real)

def test_one_model(model, device, unet, names, counts):
    ans = pd.DataFrame() # storing all the stats in a dataframe 
    model.to(device)
    model.eval()
    names = []
    means = []
    non_means = []
    dice_backs = []
    dice_befores = []
    for name in tqdm(counts.keys()):
        # iterate all names
        res = test_one_case(name, counts, model, device, unet=unet)
        names.append(name)
        means.append(res[0])
        non_means.append(res[1])
        dice_befores.append(res[2][0])
        dice_backs.append(res[2][1])
    ans['Name'] = names
    ans['Mean'] = means
    ans['Mean without backgroud'] = non_means
    ans['Dice Coefficient (label 1)'] = dice_befores
    ans['Dice Coefficient (label 0)'] = dice_backs
    return ans