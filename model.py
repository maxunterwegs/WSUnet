import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import densenet
from torchvision.models import resnet
import math
import numpy as np
from torch.nn.modules import *
from torch.nn import init
from torch.hub import load_state_dict_from_url




''' LSE-LBA pooling'''


class CustomPooling(nn.Module):
    def __init__(self, beta=1, r_0=0, dim=(-1, -2), mode=None):
        super(CustomPooling, self).__init__()
        self.r_0 = r_0

        self.dim = dim
        self.reset_parameters()
        self.mode = mode
        # self.device = device

        if self.mode is not None:
            # make a beta for each class --> size of tensor.
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(3)), requires_grad=True)
            # self.cuda(self.device)
        else:
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(1)), requires_grad=True)

    def reset_parameters(self, beta=None, r_0=None, dim=(-1, -2)):
        if beta is not None:
            init.zeros_(self.beta)
        if r_0 is not None:
            self.r_0 = r_0
        self.dim = dim

    def forward(self, x):
        '''
        :param x (tensor): tensor of shape [bs x K x h x w]
        :return logsumexp_torch (tensor): tensor of shape [bs x K], holding class scores per class
        '''

        if self.mode is None:
            const = self.r_0 + torch.exp(self.beta)
            _, _, h, w = x.shape
            average_constant = np.log(1. / (w * h))
            mod_out = const * x
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2))) / const
            return logsumexp_torch
        else:
            const = self.r_0 + torch.exp(self.beta)
            _, d, h, w = x.shape

            average_constant = np.log(1. / (w * h))
            # mod_out = torch.zeros(x.shape)
            # self.cuda(self.device)
            mod_out0 = const[0] * x[:, 0, :, :]
            mod_out1 = const[1] * x[:, 1, :, :]
            mod_out2 = const[2] * x[:, 2, :, :]

            mod_out = torch.cat((mod_out0.unsqueeze(1), mod_out1.unsqueeze(1), mod_out2.unsqueeze(1)), dim=1)
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2))) / const
            return logsumexp_torch


class MyResnetBlock(nn.Module):
    ''' create a resnet layer, with n blocks'''
    def __init__(self, block, inplanes, k, layers=1):
        super(MyResnetBlock, self).__init__()
        self.inplanes = inplanes  # dont know why yet

        self.layer1 = self._make_layer(block, k, blocks=layers, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # expansion= 1 for BasicBlock, 4 for Bottleneck
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        return x

class ModResNet(nn.Module):
    '''
    Input: Tensor: [bs x 3 x 512 x 512]
    Output: Tensor: [bs x 8*k x 8 x 8], k = 14
    '''

    def __init__(self, block, layers, num_classes=1000, k=14):    # add parameter k
        self.inplanes = 64
        super(ModResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, k, layers[0], stride=2) # adjusted for 64x downsampling, replaced planes; [64 -->k]
        self.layer2 = self._make_layer(block, 2*k, layers[1], stride=2) #  replaced planes; [128 -->2*k]
        self.layer3 = self._make_layer(block, 4*k, layers[2], stride=2) #  replaced planes; [256 -->4*k]
        self.layer4 = self._make_layer(block, 8*k, layers[3], stride=2) #  replaced planes; [512 -->8*k]
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # commented out, because final layer should be 8x8

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # expansion= 1 for BasicBlock, 4 for Bottleneck
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x) # commented out to achieve bsx8x8x8k feature space
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class MyUpsampler(nn.Module):
    def __init__(self, num_init_features, num_out_features):
        super(MyUpsampler, self).__init__()

        self.num_init_features = num_init_features
        self.num_out_features = num_out_features
        self.up = Upsample(scale_factor=2, mode='nearest') # creates [bs x num_init_features x 1/2 w, 1/2 h]

    def forward(self, x):
        return self.up(x)


# change original resnet such that it can be used pretrained
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def update_sigma(self):
        self.sigma = self.sigma * 0.999

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class ResNetPretrainable(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetPretrainable, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, full=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res1 = self.layer1(x)
        if not full:
            res1.requires_grad_().retain_grad()
        res2 = self.layer2(res1)
        if not full:
            res2.requires_grad_().retain_grad()
        res3 = self.layer3(res2)
        if not full:
            res3.requires_grad_().retain_grad()
        res4 = self.layer4(res3)
        if not full:
            res4.requires_grad_().retain_grad()

        x = self.avgpool(res4)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x, res1, res2, res3, res4


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetPretrainable(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


class ShortResnet50(ResNetPretrainable):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ShortResnet50, self).__init__(block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        res1 = self.layer1(x)
        res2 = self.layer2(res1)
        res3 = self.layer3(res2)
        res4 = self.layer4(res3)

        # x = self.avgpool(res4)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return res4


def _resnet_short(arch, block, layers, pretrained, progress, **kwargs):
    model = ShortResnet50(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_short(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_short('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

# model = resnet50_short(pretrained=True)
# input = torch.randn(1, 3, 512, 512)
# out = model(input)

# model = ModResNet(resnet.BasicBlock, [2, 2, 2, 2])
# print(model)
#
# # test model with a dummy input

# print(out)
#
# # ## try own ResnEt
# # model = MyResnetBlock(resnet.BasicBlock, layers=4)
# # print(model)

# test architecture
# model = resnet50(pretrained=True)



class SaliencyMapMaker(nn.Module):
    def __init__(self, block, layers, num_classes=1000, pretrained=True, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SaliencyMapMaker, self).__init__()

        self.num_classes = num_classes
        self.block = block
        self.layers = layers
        self.pretrained = pretrained

        self.pretrained_resnet50 = resnet50(pretrained=self.pretrained)

    def forward(self, x):
        x, res1, res2, res3, res4 = self.pretrained_resnet50(x)
        print('done')


