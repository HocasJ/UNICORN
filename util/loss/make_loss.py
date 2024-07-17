# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth

from .triplet_loss import TripletLoss

def make_loss(num_classes, ignore_index=-1):

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

    def loss_func(i2tscore, target):

        mask = target!= ignore_index
        i2tscore = i2tscore[mask]
        target = target[mask]

        if mask.sum() == 0:
            return torch.tensor([0.0]).cuda()

        I2TLOSS = xent(i2tscore, target)

        return I2TLOSS

    return loss_func


# def make_loss_clipreid(num_classes):

#     xent = CrossEntropyLabelSmooth(num_classes=num_classes)
#     triplet = TripletLoss()

#     def loss_func(feat, i2tscore, target):
    
#         TRI_LOSS = triplet(feat, target)[0]

#         I2TLOSS = xent(i2tscore, target)

#         loss =  I2TLOSS + TRI_LOSS
        
#         return loss
    
#     return loss_func
