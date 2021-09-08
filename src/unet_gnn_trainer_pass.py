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

edges = get_edges()
class Trainer():
    def __init__(self,model,train_set,test_set,opts, device):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = device
        print(self.device)
        self.model.to(self.device)
        
        self.epochs = opts['epochs']
        self.scaler = GradScaler()
        self.optim = torch.optim.AdamW(model.parameters(), opts['lr'], weight_decay=0.001)
        self.criterion = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=False)                     # loss function
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=False)
        self.tb = SummaryWriter(log_dir='./runs/unet_sageconv2/')
        self.best_loss = 0
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train() #put model in training mode
            self.tr_loss = []
            for i, (data,labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                data, labels = data.to(self.device),labels.to(self.device)
                self.optim.zero_grad()  
                outputs = self.model(data, self.device, edges)   
                loss = self.criterion(outputs, labels)
                loss.backward()                        
                self.optim.step()                  
                self.tr_loss.append(loss.item())
                self.tb.add_scalar("Train Loss", np.mean(self.tr_loss), epoch)
            
            self.test(epoch) # run through the validation set
        self.tb.close()
            
    def test(self,epoch):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_loss = []
            self.test_dice = []
            self.test_acc = []
            
            for i, (data, labels) in enumerate(self.test_loader):
                
                data, labels = data.to(self.device),labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(data, self.device, edges)
                loss = self.criterion(outputs, labels)
                self.test_loss.append(loss.item())
                outputs = torch.nn.functional.softmax(outputs, 1)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.view(-1, 1, 400, 400)
                temp_dice = compute_meandice(one_hot(predicted, 2), one_hot(labels, 2), include_background=False).detach().cpu().numpy()
                if np.nanmean(temp_dice) == np.nanmean(temp_dice):
                    self.test_dice.append(np.nanmean(temp_dice))
                self.test_acc.append((predicted == labels).sum().item() / (predicted.size(0)*400*400))
               
            print('epoch: {}, train loss: {}, test loss: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss)))
            print('epoch: {}, test dice: {}, test acc: {}'.format( 
                  epoch+1, np.nanmean(self.test_dice), np.mean(self.test_acc)))           
            self.tb.add_scalar("Val Loss", np.mean(self.test_loss), epoch)
            self.tb.add_scalar("Val dice", np.nanmean(self.test_dice), epoch)
            self.tb.add_scalar("Val acc", np.mean(self.test_acc), epoch)
            if np.nanmean(self.test_dice) > self.best_loss:
                self.best_loss = np.nanmean(self.test_dice)
                torch.save(self.model, './model_weights/best_unet_sage_conv_2.pt')