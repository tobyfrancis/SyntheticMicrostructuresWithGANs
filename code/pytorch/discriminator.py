import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorInput(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(DiscriminatorInput, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn = ContBatchNorm3d(outChans)
        self.relu = ELUCons(elu, outChans)

    def forward(self, x):
        return self.relu(elf.bn(self.conv(x)))


class ConvDown(nn.Module):
    def __init__(self, inChans, outChans, elu, dilation=1, dropout=False):
        super(ConvDown, self).__init__()
        outChans = 2*inChans
        self.dropout = dropout
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=3, stride=2)
        self.bn = ContBatchNorm3d(outChans)
        self.relu = ELUCons(elu, outChans)
        self.do = nn.Dropout3d()

    def forward(self, x):
        out = self.relu(self.bn(self.down_conv(x)))
        if self.dropout:
            out = self.do(out)
        return out

class DiscriminatorOutput(nn.Module):
    def __init__(self,inChans):
        self.conv = nn.Conv3D(inChans,16,kernel_size=3)
        self.bn = ContBatchNorm3d(outChans)
        self.relu = ELUCons(elu, outChans)
        self.dense = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.relu(self.bn(self.conv(x)))
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size()[0],out.size()[1])
        out = self.sigmoid(self.dense(out))
        return out

class Discriminator(nn.Module):
    def __init__(self, inChans, elu=True, dropout=False):
        self.input = DiscriminatorInput(inChans, 16, elu)
        self.conv_1 = ConvDown(16,32,elu)
        self.conv_2 = ConvDown(32,32,elu)
        self.conv_3 = ConvDown(32,64,elu)
        self.conv_4 = ConvDown(64,64,elu)
        self.output = DiscriminatorOutput(64)
        
    def forward(self, x):
        out = self.input(out)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.output(out)
        return out