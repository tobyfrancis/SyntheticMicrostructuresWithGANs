import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.layers import *

class GeneratorInput(nn.Module):
    #Input a set of 6*6*6*4 noise feature maps
    def __init__(self, inChans, outChans, outChans, elu):
        super(GeneratorInput, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        x = x.permute(0,4,1,2,3).continguous()
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class ConvUp(nn.Module):
    def __init__(self, inChans, outChans, elu, dropout=False):
        super(ConvUp, self).__init__()
        self.dropout = dropout
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=3, stride=2)
        self.bn = ContBatchNorm3d(outChans // 2)
        self.relu = ELUCons(elu, outChans // 2)
        self.do = nn.Dropout3d()

    def forward(self, x):
        out = self.relu1(self.bn(self.up_conv(out)))
        if self.dropout:
            out = self.do(out)
        return out

class GeneratorOutput(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(GeneratorOutput, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 8, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(8, 4, kernel_size=3)

    def forward(self, x):
        # convolve down to 4 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        return out

class Generator(nn.Module):
    def __init__(self, elu=True, nll=False):
        super(Generator, self).__init__()
        self.input = GeneratorInput(4, 64, elu) #(a*a*a*64)
        self.up_256 = ConvUp(64, 256, elu, dropout=True) #(2a*2a*2a*256)
        self.up_128 = ConvUp(256, 128, elu, dropout=True) #(4a*4a*4a*128)
        self.up_64 = ConvUp(128, 64, elu) #(8a*8a*8a*64)
        self.up_32 = ConvUp(64, 32, elu) #(16a*16a*16a*32)
        self.output = GeneratorOutput(32, elu, nll) #(16a*16a*16a*4)

    def forward(self, x):
        out = self.input(x)
        out = self.up_256(out)
        out = self.up_128(out)
        out = self.up_64(out)
        out = self.up_32(out)
        out = self.output(out)
        return out