"""
Released under BSD 3-Clause License,
Modifications are Copyright (c) 2019 Cerebras, Inc.
All rights reserved.

URL to the original source code: https://github.com/Cerebras/online-normalization

Released under BSD 3-Clause License,
Modifications are Copyright (c) 2021 Hao-Yuan Chang
All rights reserved.

== BSD 3-Clause License ==

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class UniConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(UniConv2d, self).__init__(*args, **kwargs)
        # register a buffer with the same dimension as the weight
        # this buffer will be used to store intermediate results to avoid re-exponentiation
        self.register_buffer('unitary_weight', torch.randn_like(self.weight)) 
    def forward(self, x):
        ## NOTE ##
        # For this version, assume num input channels = num output channels = n.
        # Since it's the 1x1 conv filter, we know the weight (w) have the dimesionality (n,n,1,1).
        # We will reshape w into nxn, and w records the lower triangular matrix.
        # Then, lie_algebra is constructed from w - w.transpose.
        # lie_group is constructed from matrix_exp(lie_algebra). 
        # Note the identity matrix is in matrix_exp's taylor expansion. You do not need to add it yourself.
        # Check that lie_group is an orthogonal matrix by norm(lie_group.transpose @ lie_group - idenity_matrix) ~ 0
        # Reshape lie_group back into (n,n,1,1) for convolution.
        # The weight matrix dimensions are organized as (cout,cin,filter size, filter size)
        # For the expand (cin < cout) case, the first dimension of the weight matrix will be larger than the second dim
        
        w = self.weight.flatten(start_dim=1, end_dim=-1) # restriction w has to be a square matrix, construct a 2x2 matrix from the input/output channel dimensions
        if w.shape[0] < w.shape[1]:
            transposed = True
        else:
            transposed = False
        # if we are in training mode, exponentiate the Lie parameters
        if self.training:
            if transposed:
                w = w.transpose(-2, -1)
            n = w.shape[0]
            k = w.shape[1]
            triangular = torch.zeros(n,n).to(self.weight.device) # Building the lower triangular parameter matrix
            triangular[:,0:k] = w # Assign the weight matrix to the first k columns 
            triangular = triangular.tril(-1) # only use the lower triangular portion of the weight matrix #TODO: do not waste params
            lie_algebra = triangular - triangular.transpose(-2, -1)
            lie_group = torch.matrix_exp(lie_algebra) # exponentiation 
            if transposed: # this means we are contracting dims
                unitary_weight = lie_group[:,0:k].transpose(-2, -1).reshape(self.weight.shape) # a nxk matrix containing k orthonormal basis vectors
            else:
                unitary_weight = lie_group[:,0:k].reshape(self.weight.shape) # a nxk matrix containing k orthonormal basis vectors
            # store a copy of the unitary_weight in the registered_buffer
            self.unitary_weight = unitary_weight
            if 0: # checking unitarity
                identity = torch.eye(n).to(self.weight.device)
                error = lie_group.transpose(-2, -1) @ lie_group - identity
                norm = torch.norm(error)
                assert norm < 1 # make sure the L2 norm of the error is less than 1 (an arbitary threshold)
        # otherwise, in evaluation mode, we load the unitary_weights from register_buffer
        else:
            unitary_weight = self.unitary_weight

        conv_result = F.conv2d(x, unitary_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if transposed:
            norm = torch.norm(conv_result,dim=1,keepdim=True)
            conv_result = conv_result / (norm+torch.finfo(torch.float32).eps) # normalize per pixel and filter

        return conv_result

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, **kwargs):
    if kwargs['mode'] == 'unitary':
        return UniConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
    else:
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, **kwargs):
    if kwargs['mode'] == 'unitary':
        return UniConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True, **kwargs):
    if kwargs['mode'] == 'unitary':
        return UniConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, norm_kwargs={}):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, **norm_kwargs)
        self.bn1 = norm_layer(planes, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, **norm_kwargs)
        self.bn2 = norm_layer(planes, **norm_kwargs)
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
                 base_width=64, dilation=1, norm_layer=None, norm_kwargs={}):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, **norm_kwargs)
        self.bn1 = norm_layer(width, **norm_kwargs)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, **norm_kwargs)
        self.bn2 = norm_layer(width, **norm_kwargs)
        self.conv3 = conv1x1(width, planes * self.expansion, **norm_kwargs)
        self.bn3 = norm_layer(planes * self.expansion, **norm_kwargs)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, norm_kwargs={},
                 cifar=False, kernel_size=7, stride=2, padding=3, inplanes=64):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.cifar = cifar
        self._norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs

        self.inplanes = inplanes
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
        self.conv1 = conv(3, self.inplanes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False, **norm_kwargs)
        self.bn1 = norm_layer(self.inplanes, **norm_kwargs)
        self.relu = nn.ReLU(inplace=True)
        if not cifar:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inplanes = self.inplanes
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        inplanes *= 2
        self.layer2 = self._make_layer(block, inplanes, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        inplanes *= 2
        self.layer3 = self._make_layer(block, inplanes, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if not cifar:
            inplanes *= 2
            self.layer4 = self._make_layer(block, inplanes, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
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
                conv1x1(self.inplanes, planes * block.expansion, stride, **self.norm_kwargs),
                norm_layer(planes * block.expansion, **self.norm_kwargs),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation,
                            norm_layer, norm_kwargs=self.norm_kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.cifar:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnetD(depth, **kwargs):
    """
    Constructs a ResNet model of depth D using basic blocks for cifar training
    """
    assert depth % 6 == 2, 'depth must be such that depth mod 6 = 2'
    layer_depth = (depth - 2) // 6
    return ResNet(BasicBlock, [layer_depth] * 3, **kwargs)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet44(pretrained=False, **kwargs):
    """Constructs a ResNet-44 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet92(pretrained=False, **kwargs):
    """Constructs a ResNet-92 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet143(pretrained=False, **kwargs):
    """Constructs a ResNet-143 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model
