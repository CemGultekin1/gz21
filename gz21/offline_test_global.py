import mlflow
import xarray as xr
import matplotlib.pyplot as plt
from data.utils import load_training_datasets
import os
import xarray as xr
import numpy as np
import math

raw_data = xr.open_zarr('/scratch/cz3056/CNN_train/Arthur_model/gz21/mlruns/468199358083462719/068a2b559397464e863647b59bd0a611/artifacts/forcing.zarr')
# raw_datasets = load_training_datasets(raw_data, 'training_subdomains.yaml')
print(raw_data)

low_rez = raw_datasets[0]
u = low_rez['usurf']
v = low_rez['vsurf']

import torch
import importlib
#load the neural network
def load_model_cls(model_module_name: str, model_cls_name: str):
    try:
        module = importlib.import_module(model_module_name)
        model_cls = getattr(module, model_cls_name)
    except ModuleNotFoundError as e:
        raise type(e)('Could not retrieve the module in which the trained model \
                      is defined: ' + str(e))
    except AttributeError as e:
        raise type(e)('Could not retrieve the model\'s class. ' + str(e))
    return model_cls
def load_paper_net(device: str = 'gpu'):
    """
        Load the neural network from the paper
    """
    print('In load_paper_net()')
    model_module_name = 'models.models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2, 4)
    print('After net')
    if device == 'cpu':
        transformation = torch.load('/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
        print('After torch.load()')
    else:
        transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/best_trained_model_masks.pth'
    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(model_file))
    print(net)
    return net
net = load_paper_net('cpu')