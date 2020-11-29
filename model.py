import math
import torch
import torch.nn as nn


class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        self.BN1 = nn.BatchNorm2d(channels)
        self.BN2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.BN1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.BN2(residual)

        return x + residual


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64,
         kernel_size=9, stride=1, padding=4)
        self.block1 = Residual_Block(64)
        self.block2 = Residual_Block(64)
        self.block3 = Residual_Block(64)
        self.block4 = Residual_Block(64)
        self.block5 = Residual_Block(64)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.BN64 = nn.BatchNorm2d(64)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        block0 = self.prelu(self.conv_in(x))
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.BN64(self.conv6(block5))
        block7 = self.prelu(self.pixelshuffle(self.conv7_1(block6+block0)))
        block8 = self.prelu(self.pixelshuffle(self.conv7_2(block7)))
        block9 = self.conv_out(block8)

        return (torch.tanh(block9) + 1) * 0.5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.BN64 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.BN128 = nn.BatchNorm2d(128)
        self.conv_5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.BN256 = nn.BatchNorm2d(256)
        self.conv_7 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.BN512 = nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.Dense = nn.AdaptiveAvgPool2d(1)
        self.conv_9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv_10 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):

        batch_size = x.size(0)

        out1 = self.leakyrelu(self.conv_1(x))
        out2 = self.leakyrelu(self.BN64(self.conv_2(out1)))
        out3 = self.leakyrelu(self.BN128(self.conv_3(out2)))
        out4 = self.leakyrelu(self.BN128(self.conv_4(out3)))
        out5 = self.leakyrelu(self.BN256(self.conv_5(out4)))
        out6 = self.leakyrelu(self.BN256(self.conv_6(out5)))
        out7 = self.leakyrelu(self.BN512(self.conv_7(out6)))
        out8 = self.leakyrelu(self.BN512(self.conv_8(out7)))
        out9 = self.leakyrelu(self.conv_9(self.Dense(out8)))
        out10 = self.leakyrelu(self.conv_10(out9))
        out11 = torch.sigmoid(out10.view(batch_size))

        return out11