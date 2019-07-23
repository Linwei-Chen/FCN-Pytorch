# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn


# FocalLoss的讲解
# https://zhuanlan.zhihu.com/p/32423092
# http://skyhigh233.com/blog/2018/04/04/focalloss/
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        logp = self.ce(input, target)
        print(f'logp:{logp}')
        p = torch.exp(-logp)
        print(f'p:{p}')
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()


if __name__ == '__main__':
    # test
    x = torch.Tensor([[0.1, 0.9]]).float()
    y = torch.Tensor([[0, 1]]).float()
    loss = torch.nn.BCELoss(reduction='mean')(input=x, target=y)
    focal_loss = FocalLoss()(input=x, target=y)
    print(f'loss:{loss.item()}\nfocal_loss:{focal_loss.item()}')
