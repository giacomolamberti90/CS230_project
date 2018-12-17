from util import *
from dataStructures import *
import trainer_conv2D

hparams_ANN = {
    'epochs': 300,
    'lr': 0.01,
    'layers': 5,
    'units': 7,    
    'activation': 'relu',
    'dropout': 1
}

hparams_CNN = {
    'epochs': 1000,
    'lr': 0.01,
    'layers': 1,
    'activation': 'relu',
    'dropout': 1,
    'batchNorm': 0,
    'filters': 14,
    'conv_size': 1,
    'pool_size': 1
}

#hparams_CNN = {}

if __name__ == '__main__':
    
    opt_hparams = trainer_conv2D.trainer(hparams_CNN)
