# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 23:13:41 2020

@author: Admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
# Create a class for a block 
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, expansion = 1, stride = 1):
        super(block, self).__init__()
        self.expanded_outchannels = out_channels * expansion # Borrowing the idea of expansion factor from Resnet50-101-152
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        
        self.conv2 = nn.Conv2d(out_channels, self.expanded_outchannels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.expanded_outchannels)
        
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        # print("identity.shape = ", identity.shape)
        # print("x.shape = ", x.shape)
        x += identity
        x = F.relu(x)
        return x
class ResNetCIFAR(nn.Module):
    def __init__(self,image_channels, num_classes, expansion, num_blocks_per_layer = 2):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 16 # meaning this is the first number of channels to upsample to from image_channels
        self.expansion = expansion
        self.num_blocks_per_layer = 2
        
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(16) # 
        
        # Resnet layers
        self.layerconv2 = self._make_layer(block, out_channels = 16, stride = 1)
        self.layerconv3 = self._make_layer(block, out_channels = 32, stride = 2)
        self.layerconv4 = self._make_layer(block, out_channels = 64, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * self.expansion, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print("Output shape after conv1: ", x.shape)
        x = self.layerconv2(x)
        x = self.layerconv3(x)
        x = self.layerconv4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
    
        return x
    def _make_layer(self, block, out_channels, stride):
        layers = []
        # Because any time this function _make_layer is called, this resulting layer will downsample the input.
        # Hence identity_downsample is always needed
        identity_downsample = nn.Sequential(
                                        nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size = 1,
                                                 stride = stride),
                                        nn.BatchNorm2d(out_channels * self.expansion)
                                        )
        # Creating first block for this layer
        layers.append(block(in_channels = self.in_channels, out_channels = out_channels, identity_downsample = identity_downsample, \
                            expansion = self.expansion, stride = stride))
        
        self.in_channels = out_channels * self.expansion
        
        # Creating subsequent blocks for this layer
        # For subsequent blocks, the input dimensions match the output dimensions, so no identity_downsample is needed,
        # meaning only perform simple addition of the input and the output
        for i in range(self.num_blocks_per_layer - 1):
            layers.append(block(self.in_channels, out_channels, expansion = self.expansion))
        
        return nn.Sequential(*layers)
    
# resnet = ResNetCIFAR(image_channels = 3, num_classes = 100, expansion = 3, num_blocks_per_layer = 2)
# x = torch.randn(3, 3, 32, 32)
# y = resnet(x)