#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:14:04 2024

@author: dagi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Encoder
        self.enc0 = ConvBlock(in_channels, 32)
        self.enc1 = ConvBlock(32, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(1024, 2048)

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride = 2)
        self.att5 = AttentionBlock(F_g=1024, F_l = 1024, F_int = 512)
        self.dec5 = ConvBlock(2048, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)
        
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att0 = AttentionBlock(F_g=32, F_l=32, F_int=32)
        self.dec0 = ConvBlock(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Bottleneck
        b = self.bottleneck(self.pool(e5))

        # Decoder
        d5 = self.upconv5(b)
        e5 = self.att5(d5, e5)
        d5 = self.dec5(torch.cat((d5, e5), dim=1))
        
        d4 = self.upconv4(d5)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat((d4, e4), dim=1))

        d3 = self.upconv3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))

        d2 = self.upconv2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.upconv1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        d0 = self.upconv0(d1)
        e0 = self.att0(d0, e0)
        d0 = self.dec0(torch.cat((d0, e0), dim=1))

        # Output
        out = self.out_conv(d0)
        return out

