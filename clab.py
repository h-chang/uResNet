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

##
# Delete existing log files ##
##
import os
import glob
def deleleFile(fileregex):
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(fileregex)
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

##
# HDF5 log file interface ##
##
import h5py, sys, json
class H5Logger:
    # filepath is the full path to the log file
    # the old file will be deleted if clean is set to True
    # args are the the command line aruguments
    def __init__(self,filepath,clean=True,args=None,notes=None):
        if clean:
            # this will replace the existing log file
            self.file = h5py.File(filepath, 'w')
        else:
            # this will append to the existing log file
            self.file = h5py.File(filepath, 'a')
        # record the simulation settings as an attribute to the root 
        # convert the settings into a JSON string, ignore any non serializable ones
        self.file.attrs['commandLine'] = json.dumps(sys.argv)
        self.file.attrs['hyperParam'] = json.dumps(vars(args), default=lambda o: '<not serializable>')
        self.file.attrs['notes'] = notes

        
    # write record to file
    # record is a array of tuples with [("name",value),("name",value)]
    def log(self,record,step):
        for name, value in record: 
            try:
                # try opening a dataset with the same name
                dataset = self.file['/'+name]
            except KeyError:
                # if the dataset doesn't exist, create a floating point unlimited dataset two columns
                dataset=self.file.create_dataset(name, (0, 2), maxshape=(None, 2))
            # append the new data as a new row
            curLen = dataset.len()
            dataset.resize(curLen+1,axis=0)
            dataset[curLen,0] = float(step)
            dataset[curLen,1] = float(value)
    # write the activations to hdf5 files
    def activationSnapshot(self,activation):
        try:
        # try opening a dataset with the same name
            dataset = self.file['/acitvation']
        except KeyError:
        # if the dataset doesn't exist, create a floating point unlimited dataset two columns
            dataset=self.file.create_dataset('acitvation', (0, 7), maxshape=(None, 7))
        curLen = dataset.len()
        # combined data from all layers into one long array
        combined = []
        for key,value in activation.items():
            combined += value
        newDataLen = len(combined)
        # expand and assign the new data
        dataset.resize(curLen+newDataLen,axis=0)
        dataset[curLen:,:] = combined
        # record header
        dataset.attrs['headers']=['epoch','training','layerID','min','max','mean','std']

    def close(self):
        self.file.close()
##
# Print the contents of a hdf5 file ##
##
def h5Dump(filepath):
    with h5py.File(filepath, "r") as file:
        # List all datasets
        print("\n=== Datasets: %s ===" % file.keys())
        for key in file.keys():
            print(f"\n*** Dataset: {key} ***")
            data=file[key][:]
            print(data)
        
        # List of attributes
        print("\n=== Attributes: %s ===" % list(file.attrs))
        for attribute in file.attrs:
            print(f"\n*** Attribute: {attribute} ***")
            print(file.attrs[attribute])

##
# Report simulation progress ##
##
import time
def progress(i,total,logger=None,experiment_name=""):
    if "start_time" not in globals():
        global start_time
        start_time = time.time()
    now = time.time() 
    elapsed = now - start_time
    if i > 0:
        percent_complete = i/total
        runtime = elapsed/percent_complete
        eta = start_time + runtime
        runtime_display = f"{runtime//86400:.0f}d:{runtime%86400//3600:2.0f}h:{runtime%60:2.0f}m"
        display = f"{percent_complete*100:4.1f}% done | {runtime_display} total | {time.ctime(eta)} eta | {experiment_name}"
        if logger == None:
            print(display)
        else:
            logger.info(display)

##
# Convert a string to python object references ##
##
def multi_getattr(obj, attr, default = None):
    """
    Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
    equivalent to x.a.b.c.d. When a default argument is given, it is
    returned when any attribute in the chain doesn't exist; without
    it, an exception is raised when a missing attribute is encountered.

    """
    attributes = attr.split(".")
    for i in attributes:
        try:
            obj = getattr(obj, i)
        except AttributeError:
            if default:
                return default
            else:
                raise
    return obj

##############################################################
##                       ZBB                                ##        
##############################################################
## Write to tensorboard format ##
from torch.utils.tensorboard import SummaryWriter
class TBlogger:
    def __init__(self,folderpath,clean=True):
        if clean:
            deleleFile(f'{folderpath}/events.out.tfevents.*')
        self.tbwriter = SummaryWriter(folderpath)
    # write record to file
    # record is a array of tuples with [("name",value),("name",value)]
    def log(self,record,step):
        for name, value in record: 
            self.tbwriter.add_scalar(name, value, step)
    def close(self):
        self.tbwriter.close()