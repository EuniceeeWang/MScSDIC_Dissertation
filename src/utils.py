import warnings
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

def non_zero_acc(pre, real):
    real = real.flatten()
    pre = pre.flatten()
    non_zero_real = real[((real == 1) | (pre == 1))]
    non_zero_pre = pre[((real == 1) | (pre == 1))]
    return np.sum(non_zero_real == non_zero_pre)/ len(non_zero_pre)

def np_dice(pre, real):
    k=1
    before_dice = np.sum(pre[real==k]==k)*2.0 / (np.sum(pre[pre==k]==k) + np.sum(real[real==k]==k))
    k=0
    back_dice = np.sum(pre[real==k]==k)*2.0 / (np.sum(pre[pre==k]==k) + np.sum(real[real==k]==k))
    return before_dice, back_dice

def get_edges():
    start = []
    end = []
    for i in range(625):
        if i >= 0 and i <= 24:
            if i == 0:
                start.append(0)
                end.append(1)
                start.append(0)
                end.append(50)
            elif i == 24:
                start.append(24)
                end.append(23)
                start.append(24)
                end.append(49)
            else:
                start.append(i)
                end.append(i-1)
                start.append(i)
                end.append(i+1)
                start.append(i)
                end.append(i+25)
        elif (i >= 600 and i <= 624):
            if i == 600:
                start.append(600)
                end.append(575)
                start.append(600)
                end.append(601)
            elif i == 624:
                start.append(600)
                end.append(575)
                start.append(600)
                end.append(601)
            else:
                start.append(i)
                end.append(i-1)
                start.append(i)
                end.append(i+1)
                start.append(i)
                end.append(i-25)
        elif i % 25 == 0:
            start.append(i)
            end.append(i+1)
            start.append(i)
            end.append(i+25)
            start.append(i)
            end.append(i-25)
        elif i % 25 == 24:
            start.append(i)
            end.append(i-1)
            start.append(i)
            end.append(i+25)
            start.append(i)
            end.append(i-25)     
        else:
            start.append(i)
            end.append(i-1)
            start.append(i)
            end.append(i+25)
            start.append(i)
            end.append(i-25)  
            start.append(i)
            end.append(i+1)
    edges = torch.tensor([start, end], dtype=torch.long)
    return edges