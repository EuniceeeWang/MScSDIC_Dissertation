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
import SimpleITK as sitk
import pandas as pd

info = pd.read_csv('./labeling.csv')
ct_paths = info.ct
gt_paths = info.reference
info['id'] = info.idx.apply(lambda x: x.split('/')[-1].split('.')[-3])

train_idx = np.random.choice(range(45), 36, replace=False)

train_ct = ct_paths[[x for x in range(45) if x in train_idx]]
train_gt = gt_paths[[x for x in range(45) if x in train_idx]]

val_ct = ct_paths[[x for x in range(45) if x not in train_idx]]
val_gt = gt_paths[[x for x in range(45) if x not in train_idx]]

for ct_p, gt_p in tqdm(zip(train_ct, train_gt), total=len(train_gt)):
    ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_p, sitk.sitkFloat32))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_p, sitk.sitkInt8))
    gt = zoom(gt, np.array([400, gt.shape[1], 400])/gt.shape)
    ct = zoom(ct, np.array(gt.shape)/np.array(ct.shape))
    print(gt.shape, ct.shape)
    gt[gt != 0] = 1
    print(np.unique(gt))
    ct = ct/np.mean(ct)
    for i in range(ct.shape[1]):
        hf = h5py.File('./train/{a}_{b}.h5'.format(a = ct_p.split('/')[-1].split('.')[-3], b = str(i)), 'w')
        hf.create_dataset('ct', data=ct[:, i, :])
        hf.create_dataset('gt', data=gt[:, i, :])
        
for ct_p, gt_p in tqdm(zip(val_ct, val_gt), total=len(val_gt)):
    ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_p, sitk.sitkFloat32))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_p, sitk.sitkInt8))
    gt = zoom(gt, np.array([400, gt.shape[1], 400])/gt.shape)
    ct = zoom(ct, np.array(gt.shape)/np.array(ct.shape))
    print(gt.shape, ct.shape)
    gt[gt != 0] = 1
    print(np.unique(gt))
    ct = ct/np.mean(ct)
    for i in range(ct.shape[1]):
        hf = h5py.File('./val/{a}_{b}.h5'.format(a = ct_p.split('/')[-1].split('.')[-3], b = str(i)), 'w')
        hf.create_dataset('ct', data=ct[:, i, :])
        hf.create_dataset('gt', data=gt[:, i, :])
