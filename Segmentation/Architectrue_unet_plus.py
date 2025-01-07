#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 06:58:33 2024

@author: dagi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block: two Conv2D layers + BatchNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class NestedBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(NestedBlock, self).__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, features=[ 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        super().__init__()

        self.features = features

        # Encoder
        self.encoder1 = ConvBlock(in_channels, self.features[0])
        self.encoder2 = ConvBlock(self.features[0], self.features[1])
        self.encoder3 = ConvBlock(self.features[1], self.features[2])
        self.encoder4 = ConvBlock(self.features[2], self.features[3])
        self.encoder5 = ConvBlock(self.features[3], self.features[4])
        self.encoder6 = ConvBlock(self.features[4], self.features[5])
        self.encoder7 = ConvBlock(self.features[5], self.features[6])
        self.encoder8 = ConvBlock(self.features[6], self.features[7])
        self.encoder9 = ConvBlock(self.features[7], self.features[8])
        # self.encoder10 = ConvBlock(self.features[8], self.features[9])
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2)

        # Decoder
        
        # self.up9 = nn.ConvTranspose2d(self.features[9], self.features[8], kernel_size=2, stride=2)
        # self.dec9 = NestedBlock(self.features[8], self.features[8], self.features[8])

        self.up8 = nn.ConvTranspose2d(self.features[8], self.features[7], kernel_size=2, stride=2)
        self.dec8 = NestedBlock(self.features[7], self.features[7], self.features[7])

        self.up7 = nn.ConvTranspose2d(self.features[7], self.features[6], kernel_size=2, stride=2)
        self.dec7 = NestedBlock(self.features[6], self.features[6], self.features[6])
        
        self.up6 = nn.ConvTranspose2d(self.features[6], self.features[5], kernel_size=2, stride=2)
        self.dec6 = NestedBlock(self.features[5], self.features[5], self.features[5])

        self.up5 = nn.ConvTranspose2d(self.features[5], self.features[4], kernel_size=2, stride=2)
        self.dec5 = NestedBlock(self.features[4], self.features[4], self.features[4])

        self.up4 = nn.ConvTranspose2d(self.features[4], self.features[3], kernel_size=2, stride=2)
        self.dec4 = NestedBlock(self.features[3], self.features[3], self.features[3])

        self.up3 = nn.ConvTranspose2d(self.features[3], self.features[2], kernel_size=2, stride=2)
        self.dec3 = NestedBlock(self.features[2], self.features[2], self.features[2])

        self.up2 = nn.ConvTranspose2d(self.features[2], self.features[1], kernel_size=2, stride=2)
        self.dec2 = NestedBlock(self.features[1], self.features[1], self.features[1])

        self.up1 = nn.ConvTranspose2d(self.features[1], self.features[0], kernel_size=2, stride=2)
        self.dec1 = NestedBlock(self.features[0], self.features[0], self.features[0])

        # Final output layer
        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.encoder1(x)
        p1 = self.pool(c1)
        
        c2 = self.encoder2(p1)
        p2 = self.pool(c2)

        c3 = self.encoder3(p2)
        p3 = self.pool(c3)

        c4 = self.encoder4(p3)
        p4 = self.pool(c4)

        c5 = self.encoder5(p4)
        p5 = self.pool(c5)

        c6 = self.encoder6(p5)
        p6 = self.pool(c6)

        c7 = self.encoder7(p6)
        p7 = self.pool(c7)
        
        c8 = self.encoder8(p7)
        p8 = self.pool(c8)
        
        c9 = self.encoder9(p8)
        # p9 = self.pool(c9)
        
        # c10 = self.encoder10(p9)
        
        # Decoder
        # u9 = self.up9(c10)
        # d9 = self.dec9(u9, c9)
        
        u8 = self.up8(c9)
        d8 = self.dec8(u8, c8)
        
        u7 = self.up7(d8)
        d7 = self.dec7(u7, c7)
        
        u6 = self.up6(d7)
        d6 = self.dec6(u6, c6)
        
        u5 = self.up5(d6)
        d5 = self.dec5(u5, c5)

        u4 = self.up4(d5)
        d4 = self.dec4(u4, c4)

        u3 = self.up3(d4)
        d3 = self.dec3(u3, c3)

        u2 = self.up2(d3)
        d2 = self.dec2(u2, c2)

        u1 = self.up1(d2)
        d1 = self.dec1(u1, c1)
        

        # Final layer
        return self.final(d1)




# Print model summary
# print(model)

