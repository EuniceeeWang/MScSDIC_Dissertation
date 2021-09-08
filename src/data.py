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

class segmentation(torch.utils.data.Dataset):
    def __init__(self, paths, aug=False, train=True):
        self.paths = paths # store the paths
        self.train = train # whether training
        self.aug = aug # whether use augmentation, not used in this project
        
    def __getitem__(self, idx):
        paths = self.paths[idx]
        xs = []
        f = h5py.File(paths[1], 'r') # get the middle file
        y = f['gt'][:]
        y = torch.from_numpy(y)
        y = y.float().view(1, 400, 400) # get the ground truth
        f.close()
        
        # now iterate 3 files and get the input
        for path in paths:
            f = h5py.File(path, 'r')
            x = f['ct'][:].astype(np.float)
            x = torch.from_numpy(x)
            x = x.float().view(1, 400, 400)
            xs.append(x)  
            f.close()
        x = torch.cat(xs, dim=0)
        return x, y
        
    def __len__(self):
        return len(self.paths)

def get_dataset():
    
    base = './'
    data_path = [base + 'train/' + x for x in os.listdir(base + 'train/')] # get all training paths
    names = [x.split('/')[-1].split('_')[0] for x in data_path] # get all cases' names
    counts = pd.Series(names).value_counts().to_dict() # store the count for cases
    templete =  './train/{x}_{y}.h5' # string templete
    train_paths = []
    # iterate all cases
    for name in counts.keys():
        count = counts[name]
        # iterate three continuous 
        for i in range(1, count-2):
            temp = []
            for j in range(i-1, i+2):
                temp.append(templete.format(x=name, y=j))
            train_paths.append(temp)
            
    # same process for validation data
    base = './'
    data_path = [base + 'val/' + x for x in os.listdir(base + 'val/')]
    names = [x.split('/')[-1].split('_')[0] for x in data_path]
    counts = pd.Series(names).value_counts().to_dict()
    templete =  './val/{x}_{y}.h5'
    val_paths = []
    for name in counts.keys():
        count = counts[name]
        for i in range(1, count-2):
            temp = []
            for j in range(i-1, i+2):
                temp.append(templete.format(x=name, y=j))
            val_paths.append(temp)
    return segmentation(train_paths), segmentation(val_paths, train=False)