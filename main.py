import os

used_gpu = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from load_data import data_load, data_load_fir
from models import MODELS
from utils import *
import argparse


def create_config():
    parser = argparse.ArgumentParser(description='Basic settings for human face recognition.')
    parser.add_argument('--model', type=str, default='Vgg19', help='the kind of the model')
    parser.add_argument('--mode', type=str, default='train', help='training mode')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model')
    parser.add_argument('--first_run', type=bool, default=False, help='whether the code is run for the first time')
    parser.add_argument('--input_shape', type=tuple, default=(31, 31, 3), help='the shape of the input')
    parser.add_argument('--num_classes', type=int, default=1311, help='the number of classes')
    parser.add_argument('--mean', type=list, default=[125.307, 122.95, 113.865],
                        help='the number to substract in each channel during data procession')
    parser.add_argument('--std', type=list, default=[62.9932, 62.0887, 66.7048],
                        help='the number to devide in each channel during data precession')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    cfg = create_config()
    if cfg['first_run']:
        data = data_load_fir()
    else:
        data = data_load()

    cfg = update_config(cfg)
    model = MODELS[cfg['model']](cfg)
    data = set_data(data, cfg)

    if cfg['mode'] == 'train':
        train(data, model, cfg)
    else:
        evaluate(data, model)
