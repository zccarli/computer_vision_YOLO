import torch
import torch.nn as nn

###############################################################################
# redefine conv layer
###############################################################################
def redefine_conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def redefine_conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


##############################################################################
# The implementation of Basic Block for reference
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = redefine_conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = redefine_conv3x3(planes, planes * BasicBlock.expansion)
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


##############################################################################
# Implementation of Bottleneck Block
###############################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        ##############################################################
        # TODO: Please define your layers with the BottleNeck from the paper "Deep Residual Learning for Image Recognition"
        #
        # Note: You **must not** use the nn.Conv2d here but use **redefine_conv3x3** and **redefine_conv1x1** in this script instead
        ##############################################################

        self.bottleneck = nn.Sequential(
            redefine_conv1x1(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            redefine_conv3x3(planes, planes, stride=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            redefine_conv1x1(planes, planes * Bottleneck.expansion, stride=1),
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
       

        ###############################################################
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x

        ##############################################################
        # TODO: Please write the forward function with your defined layers
        ##############################################################
        out = self.bottleneck(x)
    
       
        ###############################################################
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
class BottleneckWithDrop(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleneckWithDrop, self).__init__()
        
        ##############################################################
        # TODO: Please define your layers with the BottleNeck from the paper "Deep Residual Learning for Image Recognition"
        #
        # Note: You **must not** use the nn.Conv2d here but use **redefine_conv3x3** and **redefine_conv1x1** in this script instead
        ##############################################################
        
        self.bottleneck = nn.Sequential(
            redefine_conv1x1(in_planes, planes, stride),
            DropBlock2d(p=0.9, block_size=7), 
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            

            redefine_conv3x3(planes, planes, stride=1),
            DropBlock2d(p=0.9, block_size=7), 
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            redefine_conv1x1(planes, planes * Bottleneck.expansion, stride=1),
            DropBlock2d(p=0.9, block_size=7), 
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
     

        ###############################################################
        self.relu = nn.ReLU(inplace=True)
        self.drop = DropBlock2d(p=0.9, block_size=7)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x

        ##############################################################
        # TODO: Please write the forward function with your defined layers
        ##############################################################
        out = self.bottleneck(x)
    
       
        ###############################################################
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = self.drop(residual)

        out += residual
        out = self.relu(out)

        return out


    
class DropBlock2d(nn.Module):
    

    def __init__(self, p: float, block_size: int, inplace: bool = False) -> None:
        super().__init__()

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    def forward(self, input):
        
        if not self.training:
            return input

        N, C, H, W = input.size()
        # compute the gamma of Bernoulli distribution
        gamma = (self.p * H * W) / ((self.block_size ** 2) * ((H - self.block_size + 1) * (W - self.block_size + 1)))
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, stride=(1, 1), kernel_size=(self.block_size, self.block_size), padding=self.block_size // 2)
        mask = 1 - mask
        normalize_scale = mask.numel() / (1e-6 + mask.sum())
        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input

