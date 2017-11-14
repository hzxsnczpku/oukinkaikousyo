import os

used_gpu = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from load_data import data_load, data_load_fir
from models import *
import argparse


def create_config():
    parser = argparse.ArgumentParser(description='Basic settings for human face recognition.')
    parser.add_argument('--model', type=str, default='Vgg19', help='the kind of the model')
    parser.add_argument('--mode', type=str, default='train', help='training mode')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model')
    parser.add_argument('--first_run', type=bool, default=False, help='whether the code is run for the first time')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    cfg = create_config()
    if cfg['first_run']:
        data = data_load_fir()
    else:
        data = data_load()
    if cfg['model'] == 'Vgg19':
        model = Vgg19(data, load_model=cfg['load_model'])
    elif cfg['model'] == 'ResNet':
        model = ResNet(data, load_model=cfg['load_model'])
    elif cfg['model'] == 'DenseNet':
        model = DenseNet(data, load_model=cfg['load_model'])
    else:
        raise NotImplementedError
    if cfg['mode'] == 'train':
        model.train()
    else:
        model.evaluate()
