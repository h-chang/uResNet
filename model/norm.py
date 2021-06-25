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
import warnings
import torch
import torch.nn as nn


class Identity(nn.Module):
    __constants__ = []

    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class LayerNorm2d(nn.Module):
    __constants__ = ['weight', 'bias']

    def __init__(self, eps=1e-05, affine=True, **kwargs):
        super(LayerNorm2d, self).__init__()

        self.eps = eps
        self.affine = False # turn off affine becuase initialization at run time never worked
        self.weight = None
        self.bias = None

    def forward(self, x):
        if self.affine and self.weight is None and self.bias is None:
            self.init_affine(x)
        return nn.functional.layer_norm(x, x.shape[1:],
                                        weight=self.weight, bias=self.bias,
                                        eps=self.eps)

    def init_affine(self, x):
        # Unlike Batch Normalization and Instance Normalization, which applies
        # scalar scale and bias for each entire channel/plane with the affine
        # option, Layer Normalization applies per-element scale and bias
        _, C, H, W = x.shape
        s = [C, H, W]
        if self.affine:
            self.weight = nn.Parameter(torch.ones(s), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(s), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)


def norm(num_features, mode='batch', eps=1e-05, momentum=0.1, affine=True,
         track_running_stats=True, gn_num_groups=32,
         batch_size=None, alpha_fwd=0.999, alpha_bkw=0.99,
         ecm='ls', ls_eps=1e-05, clamp_val=5, **kwargs):
    """
    Function which instantiates a normalization scheme based on mode

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        mode: Option to select normalization method (Default: batch)
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters (weight & bias). Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Argument valid when
            using batch norm. Default: ``True``
        gn_num_groups: number of groups used in GN.
    
    OnlineNorm Args:
        alpha_fwd: the decay factor to be used in fprop to update statistics.
            Default: 0.999
        alpha_bkw: the decay factor to be used in fprop to control the gradients
            propagating through the network. Default: 0.99
        ecm: a string which defines the error compensation mechanism in OnlineNorm.
            Choice: `ac` (Activation Clamping) | `ls` (Layer Scaling).
            Default: ls
        ls_eps: if ecm is `ls`, this is the `ls` eps. Default: 1e-05
        clamp_val: if ecm is `ac` this is the clamp value. Default: 5
    """
    
    if mode == 'batch':
        warnings.warn('Normalizer: Batch')
        normalizer = nn.BatchNorm2d(num_features=num_features, eps=eps,
                                    momentum=momentum, affine=affine,
                                    track_running_stats=track_running_stats)

    elif mode == 'group':
        warnings.warn('Normalizer: Group')
        normalizer = nn.GroupNorm(gn_num_groups, num_features,
                                  eps=eps, affine=affine)

    elif mode == 'layer':
        warnings.warn('Normalizer: Layer')
        normalizer = LayerNorm2d(eps=eps, affine=affine)

    elif mode == 'instance':
        warnings.warn('Normalizer: Instance')
        normalizer = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)

    elif mode == 'unitary':
        warnings.warn('Normalizer: Unitary')
        normalizer = Identity()

    elif mode == 'none' or mode is None:
        warnings.warn('Normalizer: None')
        normalizer = Identity()
    
    else:
        raise KeyError('mode options include: "batch" | "group" | "layer" | '
                       '"instance" | "online" | "none"')

    return normalizer
