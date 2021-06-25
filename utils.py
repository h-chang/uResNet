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
import os
import sys
import time
import shutil

import torch
from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

from model import model as models
from model.norm import norm as norm_layer

import clab
import re

best_acc1 = 0

def main_worker(train_loader, val_loader, num_classes, args, cifar=False):
    global best_acc1

    scale_lr_and_momentum(args, cifar=cifar)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.cpu == 1:
        device = torch.device('cpu')

    norm_kwargs = {'mode': args.norm_mode,
                   'alpha_fwd': args.afwd,
                   'alpha_bkw': args.abkw,
                   'ecm': args.ecm,
                   'gn_num_groups': args.gn_num_groups}
    model_kwargs = {'num_classes': num_classes,
                    'norm_layer': norm_layer,
                    'norm_kwargs': norm_kwargs,
                    'cifar': cifar,
                    'kernel_size': 3 if cifar else 7,
                    'stride': 1 if cifar else 2,
                    'padding': 1 if cifar else 3,
                    'inplanes': 16 if cifar else 64}
    # if the user has specified a model, use it instead
    if args.model != '': 
        args.arch = args.model
    elif cifar:
        model_kwargs['depth'] = args.depth
        args.arch = 'resnetD'
    

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True,
                                           **model_kwargs).to(device)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](**model_kwargs).to(device)

    if args.snapshot:
        # activation is a dictionary (since we want exact mapping of layerIDs as keys & python can't find
        # it somehow if it's an array)
        # it records the stats of the activation during both training and evaluation modes
        # all the convolution layers in the model will be labeled as 0...n
        # for each entry, the value is a tuple with these elements:
        # (epoch, training mode?, min value of this layer, max, mean, std)
        activation = {}
        # this is a hook, which will be called at every forward pass
        def get_activation(layerID):
            def hook(model, input, output):
                activation[layerID] += [(epoch,
                                        model.training,
                                        layerID,
                                        float(output.min().detach().data), 
                                        float(output.max().detach().data), 
                                        float(output.mean().detach().data), 
                                        float(output.std().detach().data))]
            return hook

        layerID = 0
        # loop through all the layers in the model
        # layer_index labels all layers from top to bottom including non-conv layers
        # layerID labels all conv layers (we will only record their activations)
        for layer_index, (name, param) in enumerate(model.named_modules()):
            # search for all the conv layers
            match = re.search('.*conv.*', name)
            if match:
                # convert the name to a pointer to the actual layer module
                # i.e. convert a string to python module pointer
                layer = clab.multi_getattr(model,name)
                # initation the activation dictionary entry with an empty array 
                # for this layerID
                activation[layerID] = []
                # register the hook. This hook function will be executed after each time
                # the forward function in this layer is called
                layer.register_forward_hook(get_activation(layerID))
                # add the name of this 
                # recorded_layers+=[(layerID,name,layer_index)]
                layerID += 1

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(get_parameter_groups(model, cifar=cifar),
                                args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer,
                            milestones=args.lr_milestones,
                            gamma=args.lr_multiplier)
    logOverwrite = True
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logOverwrite = False
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False if args.seed else True
    
    if args.evaluate:
        # if the evaluate flag is on, search for the model file at sim dir
        modelpath = f"{args.logdir}/{args.name}/model_best.pth.tar"
        if os.path.isfile(modelpath):
            print("=> loading checkpoint '{}'".format(modelpath))
            checkpoint = torch.load(modelpath)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(modelpath, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(modelpath))
        # run inference with the validation dataset
        validate(val_loader, model, criterion, device, args)
        return

    # only write to log if we are in training mode
    notes = '''
        TBTime = training time for a single batch
        TDTime = training time for loading one batch of data
        TLoss = training loss
        TAcc@1 = training accuracy @1
        TAcc@5 = training accuracy @5
        VTime = valitation batch time
        VLoss = valitation loss
        VAcc@1 = valitation accuracy @ 1
        VAcc@5 = valitation accuracy @ 5
        '''
    # logger for hdf5 and tensorboard is activated only in training mode.
    if not args.evaluate:
        logger = clab.H5Logger(f"{args.logdir}/{args.name}/measurements.h5",clean=logOverwrite,args=args,notes=notes)
        tblogger = clab.TBlogger(f"{args.logdir}/{args.name}",clean=logOverwrite) # this is a directory
    
    for epoch in range(args.start_epoch, args.epochs):
        if epoch: scheduler.step()
        print(model)

        # train for one epoch
        testRecord=train(train_loader, model, criterion, optimizer, epoch, device, args)
        if not args.evaluate:
            tblogger.log(testRecord,epoch) # write record to tensorboard
            logger.log(testRecord,epoch)
        # evaluate on validation set
        acc1,valRecord = validate(val_loader, model, criterion, device, args)
        if not args.evaluate:
            tblogger.log(valRecord,epoch) # write record to tensorboard
            logger.log(valRecord,epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, args)
        if args.snapshot and not args.evaluate:
            # logging the activations of the last layer to tensorboard
            layerID = sorted(activation.keys())[-1]
            # ['epoch','training','layerID','min','max','mean','std']
            #  0         1          2          3      4     5    6
            entry=activation[layerID][-1]
            tblogger.log([('Activation @ last layer (min)',entry[3])],epoch)
            tblogger.log([('Activation @ last layer (avg)',entry[5])],epoch)
            tblogger.log([('Activation @ last layer (max)',entry[4])],epoch)
            tblogger.log([('Activation @ last layer (std)',entry[6])],epoch)

            # logging the statistics of the weights to tensorboard
            maximums, averages, minimums = [],[],[]
            for param in model.parameters():
                maximums += [param.max()]
                averages += [param.mean()]
                minimums += [param.min()]
            maximum=torch.stack(maximums).max()
            average=torch.stack(averages).mean()
            minimum=torch.stack(minimums).min()
            tblogger.log([('All layer param (max)',maximum)],epoch)
            tblogger.log([('All layer param (avg)',average)],epoch)
            tblogger.log([('All layer param (min)',minimum)],epoch)

    if not args.evaluate:
        logger.close()
        tblogger.close()
        if args.snapshot:
            logger.activationSnapshot(activation)
    
        
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    logfile = f"{args.logdir}/{args.name}/train.log"
    batch_time = AverageMeter('TBTime', ':6.3f') # training time for a single batch
    data_time = AverageMeter('TDTime', ':6.3f') # training time for loading one batch of data
    losses = AverageMeter('TLoss', ':.4e') # training loss
    top1 = AverageMeter('TAcc@1', ':6.2f') # training accuracy @1
    top5 = AverageMeter('TAcc@5', ':6.2f') # training accuracy @5
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch),logfile=logfile)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5, = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i,args)

    return [(meter.name, meter.avg) for meter in progress.meters]

def validate(val_loader, model, criterion, device, args):
    logfile = f"{args.logdir}/{args.name}/test.log"
    batch_time = AverageMeter('VTime', ':6.3f') # valitation batch time
    losses = AverageMeter('VLoss', ':.4e') # valitation loss
    top1 = AverageMeter('VAcc@1', ':6.2f') # valitation accuracy @ 1
    top5 = AverageMeter('VAcc@5', ':6.2f') # valitation accuracy @ 5
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ',logfile=logfile)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5, = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i,args)

        # TODO: this should also be done with the ProgressMeter
        with open(logfile,'a+') as f:
            f.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5)+'\n')
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, [(meter.name, meter.avg) for meter in progress.meters]
    


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    foldername = f"{args.logdir}/{args.name}"
    model_fname = os.path.join(foldername, filename)
    torch.save(state, model_fname)
    if is_best:
        shutil.copyfile(model_fname,
                        os.path.join(foldername, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix="",logfile):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile

    def print(self, batch,args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        
        with open(self.logfile,'a+') as f:
            f.write('\t'.join(entries)+'\n')
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values
    of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_parameter_groups(model, norm_weight_decay=0, cifar=False):
    """
    Separate model parameters from scale and bias parameters following norm if
    training imagenet
    """
    if cifar:
        return model.parameters()

    model_params = []
    norm_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'fc' not in name and ('norm' in name or 'bias' in name):
                norm_params += [p]
            else:
                model_params += [p]

    return [{'params': model_params},
            {'params': norm_params,
             'weight_decay': norm_weight_decay}]


def scale_lr_and_momentum(args, cifar=False, skip=False):
    """
    Scale hyperparameters given the adjusted batch_size from input
    hyperparameters and batch size

    Arguements:
        args: holds the script arguments
        cifar: boolean if we are training imagenet or cifar
        skip: boolean skipping the hyperparameter scaling.

    """
    if skip:
        return args

    print('=> adjusting learning rate and momentum. '
          f'Original lr: {args.lr}, Original momentum: {args.momentum}')

    std_b_size = 128 if cifar else 256
    
    old_momentum = args.momentum
    args.momentum = old_momentum ** (args.batch_size / std_b_size)
    args.lr = args.lr * (args.batch_size / std_b_size *
                         (1 - args.momentum) / (1 - old_momentum))

    print(f'lr adjusted to: {args.lr}, momentum adjusted to: {args.momentum}')

    return args

