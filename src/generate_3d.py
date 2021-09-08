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
import SimpleITK as sitk
from src.utils import one_hot, non_zero_acc, np_dice, get_edges
from src.data import get_dataset
edges = get_edges
templete =  './val/{x}_{y}.h5'

def save_one_case(name, counts, model, device, unet=False):
    count = counts[name]
    real = [] # use this to store ground truth
    out = [] # for storing model's output
    inp = [] # for storing model's input
    
    for i in range(1, count-2):
        paths = []
        # get all the paths
        for j in range(i-1, i+2):
            paths.append(templete.format(x=name, y=j))
            
        # same process with what we did in the data loader
        xs = []
        f = h5py.File(paths[1], 'r')
        y = f['gt'][:]
        real.append(y)
        inp.append(f['ct'][:])
        f.close()
        # iterate all path and get the output for each one, store the input and output
        for path in paths:
            f = h5py.File(path, 'r')
            x = f['ct'][:].astype(np.float)
            x = torch.from_numpy(x)
            x = x.float().view(1, 400, 400)
            xs.append(x)  
            f.close()
        x = torch.cat(xs, dim=0).view(1, 3, 400, 400)
        x = x.to(device)
        
        # since gnn models need edges information, we need to seperate them
        if unet:
            outputs = model(x)
        else:
            outputs = model(x, device, edges)
        outputs = torch.nn.functional.softmax(outputs, 1)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.view(400, 400).detach().cpu().numpy()
        out.append(predicted)
        
    out = np.array(out)
    real = np.array(real)
    inp = np.array(inp)
    
    # binary mask is int format
    out = np.uint8(out)
    real = np.uint8(real)
    
    print(out.shape, inp.shape, real.shape)
    
    
    return inp, real, out


def test_one_model(model, device, unet, model_name, names, counts):
    model.to(device) # put to device
    model.eval() # switch to evaluation mode
    for name in tqdm(counts.keys()):
        inp, real, out = save_one_case(name, counts, model, device, unet=unet)
        # create the folder for storing
        os.makedirs('./3dtestoutput/{x}/'.format(x = name), exist_ok=True)
        
        # for all three models, using the test_one_case function to store the output of this model's prediction
        # we only store the ground truth once which is when urnning the unet
        if model_name=='unet':
            
            img_nii = sitk.GetImageFromArray(real)
            loc =  './3dtestoutput/{x}/'.format(x = name) + 'grount_truth.nii.gz'
            sitk.WriteImage(img_nii, loc) 
            
            img_nii = sitk.GetImageFromArray(out)
            loc =  './3dtestoutput/{x}/'.format(x = name) + 'unet.nii.gz'
            sitk.WriteImage(img_nii, loc)
            
        elif model_name=='linear':
            img_nii = sitk.GetImageFromArray(out)
            loc =  './3dtestoutput/{x}/'.format(x = name) + 'gnn_unet_linear.nii.gz'
            sitk.WriteImage(img_nii, loc)    
        
        else:
            img_nii = sitk.GetImageFromArray(out)
            loc =  './3dtestoutput/{x}/'.format(x = name) + 'gnn_unet_pass.nii.gz'
            sitk.WriteImage(img_nii, loc)