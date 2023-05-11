import torch
import numpy as np
import importlib
import math
import time
import pickle
import os
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
    model_module_name = 'gz21.models.models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2, 4)
    tempfile = 'tmpjbnb2092'
    root = f'/scratch/cg3306/climate/temp/gz21/temp/{tempfile}/models'
    print('After net')
    transform_file = f'{root}/transformation'
    with open(transform_file,'rb') as f:
        transformation = pickle.load(f)#map_location = "cuda:0")
    print('After torch.load()')
    transformation._min_value.data = transformation._min_value.data.to(torch.device("cpu"))
    path = os.path.join(root,'final_transformation.pth')
    torch.save(transformation,path)
    print(path)
    return
    # else:
    #     transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/.pth'
    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu_04292023.pth'
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(model_file))
    print(net)
    return net
nn = load_paper_net('cpu')
# nn.eval()