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

import argparse
import os
import random
import warnings

import torch
from torch.backends import cudnn
from torchvision import transforms
from torchvision import datasets

from model import model as models
from utils import main_worker
import os

NUM_CLASSES = 100

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-d', '--depth', default=20, type=int, metavar='D',
                    help='depth of ResNet (default: 20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run (default: 250)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts, default: 0)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-eb', '--ebatch-size', default=128, type=int,
                    metavar='N',
                    help='evaluation mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)',
                    dest='lr')
parser.add_argument('--lr-milestones', nargs='+', type=int,
                        default=[100, 150, 200],
                        help='epochs at which we take a learning-rate step '
                             '(default: [100, 150, 200])')
parser.add_argument('--lr-multiplier', default=0.1, type=float, metavar='M',
                    help='lr multiplier at lr_milestones (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='optimizer momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 2e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
norm_choices=['batch', 'group', 'layer', 'instance', 'none','unitary']
parser.add_argument('--norm-mode', default='unitary', type=str,
                    metavar='NORM', choices=norm_choices,
                    help='norm choices: ' +
                        ' | '.join(norm_choices) +
                        ' (default: batch)')
parser.add_argument('--afwd', '--decay-factor-forward', default=1023 / 1024,
                    type=float, metavar='AFWD', dest='afwd',
                    help='forward decay factor which sets momentum process '
                         'hyperparameter when using online normalization '
                         '(default: 1023 / 1024)')
parser.add_argument('--abkw', '--decay-factor-backward', default=127 / 128,
                    type=float, metavar='ABKW', dest='abkw',
                    help='backward decay factor which sets control process '
                         'hyperparameter when using online normalization '
                         '(default: 127 / 128)')
ecm_choices=['ls', 'ac', 'none']
parser.add_argument('--ecm', default='ls', type=str,
                    metavar='ECM', choices=ecm_choices,
                    help='Online Norm ErrorCompensationMechanism choices: ' +
                        ' | '.join(ecm_choices) +
                        ' (default: ls)')
parser.add_argument('--gn-num-groups', default=8, type=int,
                    help='number of groups in group norm if using group norm '
                         'as normalization method (default: 8)')
parser.add_argument('--cpu', default=0, type=int,
                    help='force to use cpu or not. gpu will be disabled if set to 1')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument("--logdir", required=True,
                    help="Where to log training info (small).")
parser.add_argument("--name", required=True,
                    help="Name of this run. Used for monitoring and checkpointing.")
parser.add_argument('--snapshot', default=0, type=int,
                    help='whether to record the activations (0/1). default=0')
parser.add_argument('--model', default='resnet44', type=str,
                    help='select which model to use and override the depth parameter resnet44/92/143 (default: resnetD for cifar10)')
args = parser.parse_args()


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    print('=> create train dataset')
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

    train_transform = transforms.Compose([transforms.Pad(4),
                                          transforms.RandomCrop(size=32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(args.data, train=True,
                                     transform=train_transform,
                                     target_transform=None,
                                     download=True)

    print('=> create train dataloader')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    print('=> create val dataset')
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    val_dataset = datasets.CIFAR100(args.data, train=False,
                                   transform=val_transform,
                                   target_transform=None,
                                   download=True)

    print('=> create val dataloader')
    print('=> creating validation dataloader...')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.ebatch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    main_worker(train_loader, val_loader, NUM_CLASSES, args, cifar=True)


if __name__ == '__main__':
    foldername = f"{args.logdir}/{args.name}"
    os.makedirs(foldername, exist_ok=True)
    main(args)
