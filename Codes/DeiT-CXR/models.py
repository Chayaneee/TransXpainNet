#import torch
#from torchvision import models as models
#import torch.nn as nn
#import timm

import os
import sys
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,sampler
from torchvision import models


## Now, we import timm, torchvision image models
import timm
from timm.loss import LabelSmoothingCrossEntropy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def model(model_name, pretrained, requires_grad):
    
    if model_name == 'ResNet':
        model = models.resnet50(progress=True, pretrained=pretrained)
        model = model.to(device)

        # to freeze the hidden layers
        if requires_grad == False:
            for param in model.parameters():
                param.requires_grad = False
        # to train the hidden layers
        elif requires_grad == True:
            for param in model.parameters():
                param.requires_grad = True
        # make the classification layer learnable
        # we have 14 classes in total
        model.fc = nn.Linear(2048, 14)
        
        model = model.to(device)
        return model

    elif model_name == 'ViT':
        #model = torch.hub.load('facebookresearch/deit:main','deit_base_patch16_224', pretrained= True)
        model = timm.create_model('deit_base_patch16_384', pretrained=True)
        model = model.to(device)

        if requires_grad == False:
            for param in model.parameters():
                param.requires_grad = False
        # to train the hidden layers
        elif requires_grad == True:
            for param in model.blocks[:5].parameters():
                param.requires_grad = False
                
            for param in model.blocks[5:].parameters():
                param.requires_grad = True

        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 14)
        )
        model = model.to(device)

        return model 

