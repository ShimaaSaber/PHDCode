# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:37:20 2021

@author: Shimaa Saber
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.nn import init

#####################################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout > 0:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f

##############################################################################  

#  Encoder - Decoder attention
class Residual_attention_Net(nn.Module):

    def __init__(self, input_dim=2048):
        super(Residual_attention_Net, self).__init__()
        self.res_attention = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, 3),
        )

    def forward(self, x):
        return self.res_attention(x)

############################################################################## 


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
            
        return x_out

############################################################################## 

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        
        self.TripletAtt= TripletAttention(256)
        self.res_att = Residual_attention_Net(input_dim=1024)
        
    def forward(self, x):        
        c_att = self.TripletAtt(x)
        res_att = self.res_att(x)        
        x1 = x * res_att        
        x = (3/4) *  c_att + (1/4) * x1        
        return  x
    

##############################################################################    
  
class Network(nn.Module):

    def __init__(self, class_num):
        super(Network, self).__init__()
        trained=False
        baseline = models.resnet101(trained)        
        baseline.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model = baseline
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3 
        self.Attention = Attention()
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.classifier = ClassBlock(2048, class_num, dropout=0.5, relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)   
        x = self.Attention(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x, f = self.classifier(x)
        return  x
    