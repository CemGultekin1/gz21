import torch
import numpy as np
import importlib
import math
import time
from torch.nn import Parameter

REPO = 'subgrid'  #'gz21'

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

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
    model_module_name = f'{REPO}.models.models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2, 4)
    
    # final_transform= '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation_04292023.pth'
    # print('After net')
    # if device == 'cpu':
    #     transformation = torch.load(final_transform)
    #     print('After torch.load()')
    # else:
    #     transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    # net.final_transformation = transformation
    print('After transformation')
    # Load parameters of pre-trained model
    print('After mlflow.tracking.MlflowClient()')
    
    
    # ----------------- CHANGE THIS PATH TO TRAINED MODEL ----------------- #
    tempfile = 'tmptsp2jxp1'#'tmputijzpt_'#
    model_file = f'/scratch/cg3306/climate/subgrid/gz21/temp/{tempfile}/models/trained_model.pth'
    # ---------------------------------------------------- #
    
    
    print('Loading final transformation')
    model_module_name = f'{REPO}.models.transforms'
    model_cls_name = 'SoftPlusTransform'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    model_cls_name = 'PrecisionTransform'
    model_cls1 = load_model_cls(model_module_name, model_cls_name)
    transform = model_cls.__new__(model_cls,)
    model_cls1.__init__(transform,)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    transform._min_value = Parameter(state_dict.pop('final_transformation._min_value'))
    print('After download_artifacts()')
    net.load_state_dict(state_dict)
    net.final_transformation = transform
    print(net)
    return net
nn = load_paper_net('cpu')
nn.eval()