import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
            
        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias = True)
        self.bn1   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias = True)
        self.bn2   = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias = True)
        self.bn3   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias = True)
        self.bn4   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias = True)
        self.bn5   = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 1, stride, bias = True),
                            nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output += F.relu(self.shortcut(input))
        output = F.relu(output)
        return output

class BasicBlock2(nn.Module):
    def __init__(self, dim):
        super(BasicBlock2, self).__init__()
        in_channels = dim
        out_channels = dim
        reduction = 0.5
        stride = 1
        self.nfe = 0
            
        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias = True)
        self.bn1   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias = True)
        self.bn2   = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias = True)
        self.bn3   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias = True)
        self.bn4   = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias = True)
        self.bn5   = nn.BatchNorm2d(out_channels)
        
            
    def forward(self, t, x):
        self.nfe += 1
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        return output

class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes, ODEBlock_):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64
        self.ODEBlock = ODEBlock_
        
        self.conv1  = nn.Conv2d(3, int(width_x * self.in_channels), 3, 1, 1, bias=True)     # For Cifar10
        self.bn1    = nn.BatchNorm2d(int(width_x * self.in_channels))
        self.stage1_1 = self._make_layer1(1, width_x, 32, 1)
        self.stage1_2 = self._make_layer2(blocks[0] - 1, width_x, 32, 1)

        self.stage2_1 = self._make_layer1(1, width_x, 64, 2)
        self.stage2_2 = self._make_layer2(blocks[1] - 1, width_x, 64, 1)

        self.stage3_1 = self._make_layer1(1, width_x, 128, 2)
        self.stage3_2 = self._make_layer2(blocks[2] - 1, width_x, 128, 1)

        self.stage4_1 = self._make_layer1(1, width_x, 256, 2)
        self.stage4_2 = self._make_layer2(blocks[3] - 1, width_x, 256, 1)
        self.conv2  = nn.Conv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias = True)
        self.bn2    = nn.BatchNorm2d(int(width_x * 128))
        self.linear = nn.Linear(int(width_x * 128), num_classes)
        
    # with residual connection mismatch
    def _make_layer1(self, num_block, width_x, out_channels, stride):
        print("in_channels = ", self.in_channels)
        strides = [stride] + [1] * (num_block - 1)
        layers  = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_layer2(self, num_block, width_x, out_channels, stride):
        print("in_channels = ", self.in_channels)
        strides = [stride] + [1] * (num_block - 1)
        layers  = []
        for _stride in strides:
            layers.append(self.ODEBlock(BasicBlock2(int(width_x * self.in_channels))))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.stage1_1(output)
        output = self.stage1_2(output)
        output = self.stage2_1(output)
        output = self.stage2_2(output)
        output = self.stage3_1(output)
        output = self.stage3_2(output)
        output = self.stage4_1(output)
        output = self.stage4_2(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output
    
def SqNxt_23_1x(num_classes, ODEBlock):
    return SqueezeNext(1.0, [2, 2, 2, 2], num_classes, ODEBlock)
def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1
    return lr / math.pow(10, (optim_factor))

