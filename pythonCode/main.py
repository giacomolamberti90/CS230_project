from util import *
from dataStructures import *
import trainer_conv2D

hparams_ANN = {
    'epochs': 500,
    'lr': 0.01,
    'layers': 3,
    'units': 1,    
    'activation': 'tanh',
    'dropout': 1
}

hparams_CNN = {
    'epochs': 500,
    'lr': 0.01,
    'layers': 5,
    'activation': 'relu',
    'dropout': 1,
    'batchNorm': 0,
    'filters': 17,
    'conv_size': 1,
    'pool_size': 1
}

#hparams_CNN = {}

if __name__ == '__main__':
    
    opt_hparams = trainer_conv2D.trainer(hparams_CNN)
