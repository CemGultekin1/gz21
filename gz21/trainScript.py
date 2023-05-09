# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:13:28 2019

@author: Arthur
"""
import os
from typing import List
from gz21.paths import TEMP
import numpy as np
import mlflow
import os.path
import tempfile
import xarray as xr

from torch.utils.data import DataLoader,Dataset,ConcatDataset
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn
import torch.nn.functional as F

# These imports are used to create the training datasets
from data.datasets import (DatasetWithTransform, DatasetTransformer,
                           RawDataFromXrDataset, ConcatDataset_,
                           Subset_, ComposeTransforms, MultipleTimeIndices)

# Some utils functions
from train.utils import (DEVICE_TYPE, learning_rates_from_string,
                         run_ids_from_string, list_from_string)
from data.utils import load_data_from_past, load_training_datasets, load_data_from_run,find_latest_data_run
from testing.utils import create_test_dataset
from testing.metrics import MSEMetric, MaxMetric
from train.base import Trainer
import train.losses
import models.transforms

import argparse
import importlib
import pickle

from data.xrtransforms import SeasonalStdizer

import models.submodels
import sys

import copy

from utils import TaskInfo
from dask.diagnostics import ProgressBar


def negative_int(value: str):
    return -int(value)

def check_str_is_None(s: str):
    return None if s.lower() == 'none' else s


# rundict = find_latest_data_run()

# PARAMETERS ---------
description = 'Trains a model on a chosen dataset from the store. Allows \
    to set training parameters via the CLI.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--exp_id', type=str,default = 0,#rundict['experiment_id'],
                    help='Experiment id of the source dataset containing the '\
                    'training data.')
parser.add_argument('--run_id', type=str,default = 0,#rundict['run_id'],
                    help='Run id of the source dataset')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=learning_rates_from_string,
                    default='0 1e-3')#{'0\1e-3'})
parser.add_argument('--train_split', type=float, default=0.8,
                    help='Between 0 and 1')
parser.add_argument('--test_split', type=float, default=0.8,
                    help='Between 0 and 1, greater than train_split.')
parser.add_argument('--time_indices', type=negative_int, nargs='*')
parser.add_argument('--printevery', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help="Depreciated. Controls the weight decay on the linear "
                         "layer")
parser.add_argument('--model_module_name', type=str, default='models.models1',
                    help='Name of the module containing the nn model')
parser.add_argument('--model_cls_name', type=str, default='FullyCNN',
                    help='Name of the class defining the nn model')
parser.add_argument('--loss_cls_name', type=str,
                    default='HeteroskedasticGaussianLossV2',
                    help='Name of the loss function used for training.')
parser.add_argument('--transformation_cls_name', type=str,
                    default='SquareTransform',
                    help='Name of the transformation applied to outputs ' \
                    'required to be positive. Should be defined in ' \
                    'models.transforms.')
parser.add_argument('--submodel', type=str, default='transform1')
parser.add_argument('--features_transform_cls_name', type=str, default='None',
                    help='Depreciated')
parser.add_argument('--targets_transform_cls_name', type=str, default='None',
                    help='Depreciated')
parser.add_argument('--seed', type=int, default=0,
                    help='torch.manual_seed')
params = parser.parse_args()

# Log the experiment_id and run_id of the source dataset
mlflow.log_param('source.experiment_id', params.exp_id)
mlflow.log_param('source.run_id', params.run_id)

# Training parameters
# Note that we use two indices for the train/test split. This is because we
# want to avoid the time correlation to play in our favour during test.
batch_size = params.batchsize
learning_rates = params.learning_rate
weight_decay = params.weight_decay
n_epochs = params.n_epochs
train_split = params.train_split
test_split = params.test_split
model_module_name = params.model_module_name
model_cls_name = params.model_cls_name
loss_cls_name = params.loss_cls_name
transformation_cls_name = params.transformation_cls_name
# Transforms applied to the features and targets
temp = params.features_transform_cls_name
features_transform_cls_name = check_str_is_None(temp)
temp = params.targets_transform_cls_name
targets_transform_cls_name = check_str_is_None(temp)
# Submodel (for instance monthly means)
submodel = params.submodel
torch.manual_seed(params.seed)

# Parameters specific to the input data
# past specifies the indices from the past that are used for prediction
indices = params.time_indices

# Other parameters
print_loss_every = params.printevery
model_name = 'trained_model.pth'

# Directories where temporary data will be saved
data_location = tempfile.mkdtemp(dir=TEMP)
print('Created temporary dir at  ', data_location)

figures_directory = 'figures'
models_directory = 'models'
model_output_dir = 'model_output'


def _check_dir(dir_path):
    """Create the directory if it does not already exists"""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


_check_dir(os.path.join(data_location, figures_directory))
_check_dir(os.path.join(data_location, models_directory))
_check_dir(os.path.join(data_location, model_output_dir))


# Device selection. If available we use the GPU.
# TODO Allow CLI argument to select the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = DEVICE_TYPE.GPU if torch.cuda.is_available() \
                            else DEVICE_TYPE.CPU
print('Selected device type: ', device_type.value)
# FIN PARAMETERS --------------------------------------------------------------


# DATA-------------------------------------------------------------------------



def get_length(varname:str):
    global_ds =  load_data_from_past()
    ntimes = len(global_ds.time)
    global train_split,test_split
    train_index = int(train_split * ntimes)
    test_index = int(test_split * ntimes)
    if varname == "train_dataset":
        return train_index
    else:
        return ntimes - test_index + 1
def dataset_initiator():
    # Split into train and test datasets
    global_ds =  load_data_from_past()#load_data_from_run(params.run_id)
    global train_split,test_split,submodel
    datasets, train_datasets, test_datasets = list(), list(), list()
    xr_datasets :List[xr.Dataset]= load_training_datasets(global_ds, 'gz21/training_subdomains.yaml')
    for domain_id,xr_dataset in enumerate(xr_datasets):
        submodel_transform = copy.deepcopy(getattr(models.submodels, submodel))
        xr_dataset = submodel_transform.fit_transform(xr_dataset)
        dataset = RawDataFromXrDataset(xr_dataset)
        dataset.index = 'time'
        dataset.add_input('usurf')
        dataset.add_input('vsurf')
        dataset.add_output('S_x')
        dataset.add_output('S_y')
        # TODO temporary addition, should be made more general
        if submodel == 'transform2':
            dataset.add_output('S_x_d')
            dataset.add_output('S_y_d')
        train_index = int(train_split * len(dataset))
        test_index = int(test_split * len(dataset))
        features_transform = ComposeTransforms()
        targets_transform = ComposeTransforms()
        transform = DatasetTransformer(features_transform, targets_transform)
        dataset = DatasetWithTransform(dataset, transform)
        train_dataset = Subset_(dataset, np.arange(train_index))
        test_dataset = Subset_(dataset, np.arange(test_index, len(dataset)))
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        datasets.append(dataset)
    train_dataset = ConcatDataset_(train_datasets)
    test_dataset = ConcatDataset_(test_datasets)
    return train_dataset,test_dataset,train_datasets,test_datasets,datasets
class LazyDatasetWrapper(ConcatDataset_):
    def __init__(self, varname,):
        self.varname = varname
        self._lazy_init_flag = False
        self._transform_from_model_flag = False
        self._model = None
        self._length = get_length(varname)*4
        self._subset = None
    def add_transforms_from_model(self,model):
        self._model = model
    def lazy__init__(self,):
        train_dataset,test_dataset,datasets,_,_ = dataset_initiator()
        if self.varname == "train_dataset":
            subset =  train_dataset
        else:
            subset =  test_dataset
        for dataset in datasets:
            dataset.add_transforms_from_model(self._model)
        self.__dict__.update(subset.__dict__)
        self._subset = subset
    def inverse_transform_target(self,*args,**kwargs):
        if not self._lazy_init_flag:
            self.lazy__init__()
            self._lazy_init_flag = True
        return self._subset.datasets[0].inverse_transform_target(*args,**kwargs)
    def __getitem__(self,*args):
        if not self._lazy_init_flag:
            self.lazy__init__()
            self._lazy_init_flag = True
        return ConcatDataset.__getitem__(self,*args)
    def __len__(self,):
        return self._length

# Concatenate datasets. This adds shape transforms to ensure that all regions
# produce fields of the same shape, hence should be called after saving
# the transformation so that when we're going to test on another region
# this does not occur.
train_dataset = LazyDatasetWrapper('train_dataset',) #ConcatDataset_(train_datasets)
test_dataset = LazyDatasetWrapper('test_dataset',) #ConcatDataset_(test_datasets)
test_dataset_for_transform = LazyDatasetWrapper('test_dataset',) 
# FIN DATA---------------------------------------------------------------------


# NEURAL NETWORK---------------------------------------------------------------
# Load the loss class required in the script parameters
n_target_channels = 2#datasets[0].n_targets
criterion = getattr(train.losses, loss_cls_name)(n_target_channels)

# Recover the model's class, based on the corresponding CLI parameters
try:
    models_module = importlib.import_module(model_module_name)
    model_cls = getattr(models_module, model_cls_name)
except ModuleNotFoundError as e:
    raise type(e)('Could not find the specified module for : ' +
                str(e))
except AttributeError as e:
    raise type(e)('Could not find the specified model class: ' +
                str(e))
    
net = model_cls(2,4)#datasets[0].n_features, criterion.n_required_channels)
try:
    transformation_cls = getattr(models.transforms, transformation_cls_name)
    transformation = transformation_cls()
    transformation.indices = criterion.precision_indices
    net.final_transformation = transformation
except AttributeError as e:
    raise type(e)('Could not find the specified transformation class: ' +
                str(e))

print('--------------------')
print(net)
print('--------------------')
print('***')

# raise Exception
# Log the text representation of the net into a txt artifact
with open(os.path.join(data_location, models_directory,
                    'nn_architecture.txt'), 'w') as f:
    print('Writing neural net architecture into txt file.')
    f.write(str(net))
# FIN NEURAL NETWORK ---------------------------------------------------------


train_dataset.add_transforms_from_model(net)
test_dataset.add_transforms_from_model(net)



print('Size of training data: {}'.format(len(train_dataset)))
print('Size of validation data : {}'.format(len(test_dataset)))
# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True, num_workers=8)



# Training---------------------------------------------------------------------
# Adam optimizer
# To GPU
net.to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=learning_rates[0], weight_decay=weight_decay)
lr_scheduler = MultiStepLR(optimizer, list(learning_rates.keys())[1:],
                        gamma=0.1)

trainer = Trainer(net, device)
trainer.criterion = criterion
trainer.print_loss_every = print_loss_every

# metrics saved independently of the training criterion.
metrics = {'R2': MSEMetric(), 'Inf Norm': MaxMetric()}
features_transform = ComposeTransforms()
targets_transform = ComposeTransforms()
transform = DatasetTransformer(features_transform, targets_transform)
for metric_name, metric in metrics.items():
    metric.inv_transform = lambda x: transform.inverse_transform_target(x)
    trainer.register_metric(metric_name, metric)

for i_epoch in range(n_epochs):
    print('Epoch number {}.'.format(i_epoch))
    # TODO remove clipping?
    train_loss = trainer.train_for_one_epoch(train_dataloader, optimizer,
                                            lr_scheduler, clip=1.)
    test = trainer.test(test_dataloader)
    if test == 'EARLY_STOPPING':
        print(test)
        break
        
    test_loss, metrics_results = test
    # Log the training loss
    print('Train loss for this epoch is ', train_loss)
    print('Test loss for this epoch is ', test_loss)
    print('Learning rate ', optimizer.param_groups[0]['lr'])

    for metric_name, metric_value in metrics_results.items():
        print('Test {} for this epoch is {}'.format(metric_name, metric_value))
    mlflow.log_metric('train loss', train_loss, i_epoch)
    mlflow.log_metric('test loss', test_loss, i_epoch)
    mlflow.log_metrics(metrics_results)
# Update the logged number of actual training epochs
mlflow.log_param('n_epochs_actual', i_epoch + 1)

# FIN TRAINING ----------------------------------------------------------------

# Save the trained model to disk
net.cpu()
full_path = os.path.join(data_location, models_directory, model_name)
torch.save(net.state_dict(), full_path)
net.cuda(device)

# Save other parts of the model
# TODO this should not be necessary
print('Saving other parts of the model')
full_path = os.path.join(data_location, models_directory, 'transformation')
with open(full_path, 'wb') as f:
    pickle.dump(transformation, f)

with TaskInfo('Saving trained model'):
    mlflow.log_artifact(os.path.join(data_location, models_directory))

# DEBUT TEST ------------------------------------------------------------------
_,_,_,_,test_datasets = dataset_initiator()
for i_dataset, test_dataset in enumerate(test_datasets,):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=True)
    output_dataset = create_test_dataset(net, criterion.n_required_channels,test_dataset,test_dataloader, device)

    # Save model output on the test dataset
    output_dataset.to_zarr(os.path.join(data_location, model_output_dir,
                                        f'test_output{i_dataset}'))

# Log artifacts
print('Logging artifacts...')
mlflow.log_artifact(os.path.join(data_location, figures_directory))
mlflow.log_artifact(os.path.join(data_location, model_output_dir))
print('Done...')
