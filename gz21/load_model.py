import torch
import numpy as np
import importlib
import math
import time

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
    net = model_cls(2, 4)
    
    tempfile = 'tmpgbyo08g9'
    root = f'/scratch/cg3306/climate/temp/gz21/temp/{tempfile}/models'
    if device == 'gpu':
        transformation = torch.load(f'{root}/transformation', map_location=torch.device('cpu'))
        print('After torch.load()')
        net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')

    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = f'{root}/trained_model.pth'
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
    print(net)
    return net
nn = load_paper_net('cpu')
nn.eval()

x = torch.zeros((1,2,30,30),dtype = torch.float32)
y = nn(x)[0]
print(torch.mean(torch.mean(y,dim = 1),dim = 1))

'''
import torch
import mlflow
import pickle
import importlib

def pickle_artifact(run_id: str, path: str):
    client = mlflow.tracking.MlflowClient()
    file = client.download_artifacts(run_id, path)
    f = open(file, 'rb')
    return pickle.load(f)
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
    print('After net')
    # if device == 'cpu':
    #     transformation = torch.load('/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
    #     print('After torch.load()')
    # else:
    transformation = pickle_artifact('1825b5d4b59c4b879f27972d3e09e92a', 'models/transformation')
    net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')
    model_file = 'mlruns/535461833256949152/1825b5d4b59c4b879f27972d3e09e92a/artifacts/models/trained_model.pth'
    print('After download_artifacts()')
    # if device == 'cpu':
    #     print('Device: CPU')
    #     model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
    #     net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    # else:
    net.load_state_dict(torch.load(model_file))
    print(net)
    return net
nn = load_paper_net('gpu')
nn.eval()


import os
nn.cpu()

print(nn)
full_path = os.path.join('temp', 'exmple', 'trained_model.pth')
torch.save(nn.state_dict(), full_path)

full_path = os.path.join('temp', 'exmple', 'final_transformation.pth')
transformation = nn.final_transformation
# transformation._min_value.to("cpu")
# print(transformation._min_value.device)
torch.save(transformation,full_path)
# with open(full_path, 'wb') as f:
#     pickle.dump(transformation, f)'''