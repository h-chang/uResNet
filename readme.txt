"""
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

==== Enviroment Setup ====

1. Download and install anaconda
    https://docs.anaconda.com/anaconda/install/
2. Create a new conda enviroment
    conda create -n torch anaconda python=3.8
3. Install PyTorch
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
4. Install required conda packages
    conda install --file requirements.txt

==== Running the Simulation ====

1. Train a unitary neural network for image recognition by running cifar10_main.py with these minimun amount of command line parameters:
    python cifar10_main.py <PATH TO CIFAR-10 DATASET> --logdir <PATH TO LOG DIRECTORY> --name <NAME OF THE EXPERIEMENT> --gpu <THE DEVICE ID OF THE GPU> --model <DEPTH OF THE NETWORK>
Full Example:
    python cifar10_main.py ~/dataset/cifar10 --logdir ~/test --name test --gpu 0 --model resnet44
2. Other normalization type reported in the paper can be reproducted by changing the --norm-mode switch.
Possible choices: 
    ['batch', 'group', 'layer', 'instance', 'none','unitary']
