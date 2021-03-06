import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import keras.backend as K

from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from util import *
from dataStructures import *

from hyperopt import Trials, tpe,  fmin, hp, space_eval
from hyperopt.mongoexp import MongoTrials
from matplotlib import ticker

def my_loss(labels, predict):
    
    cost = K.sum(K.square(labels-predict))/K.sum(K.square(labels))
    
    return cost

def error(labels, predict):
    
    predict = np.reshape(predict, (len(predict)))
   
    err = np.sum(np.abs(labels-predict))/np.sum(np.abs(labels))

    return err

def train(hparams):
    
    print(hparams)
    np.random.seed(1)
    
    # Data --------------------------------------------------------------------
    # number of features
    
    # training images
    X_train = np.vstack((X_A0, X_A180, X_B0, X_B180, X_A10, X_B10, X_A170, X_B170, X_A260, X_B190))
    Y_train = np.vstack((Y_A0, Y_A180, Y_B0, Y_B180, Y_A10, Y_B10, Y_A170, Y_B170, Y_A260, Y_B190))
      
    # dev images
    X_dev = np.vstack((X_B0, X_B180, X_A190, X_B260))
    Y_dev = np.vstack((Y_B0, Y_B180, Y_A190, Y_B260))
     
    # Model -------------------------------------------------------------------
    def model(input_shape):
    
        X_input = Input(shape=input_shape)
        
        # first layer
        X = X_input
        
        # hidden layers
        for i in range(hparams['layers']):
            
            X = Conv2D(filters=hparams['filters'], kernel_size=hparams['conv_size'],
                       strides=(1,1), padding='same', activation=hparams['activation'])(X)
            
            #X = MaxPooling2D(pool_size=hparams['pool_size'], padding='same')(X)
            
            if hparams['dropout'] != 1:    
                X = Dropout(rate=hparams['dropout'])(X)
                
            if hparams['batchNorm'] == 1:
                X = BatchNormalization()(X)
            
        # output layer
        X = Conv2D(filters=1, kernel_size=hparams['conv_size'], strides=(1, 1), 
                   padding='same', activation='sigmoid')(X)
                
        # model instance
        model = Model(inputs=X_input, outputs=X)
        
        return model
    
    # Model instance ----------------------------------------------------------
    keras.backend.clear_session()
    tf.reset_default_graph()
    model = model(input_shape=(None, None, n))
    
    # Evaluate model ----------------------------------------------------------
    def eval_model():
        
        ## optimization
        opt = Adam(lr=hparams['lr'])
        model.compile(loss='mse', optimizer=opt)
        
        # train
        np.random.seed(1)
        model.fit(X_train, Y_train, epochs=hparams['epochs'])
        
        # save model
        #model.save_weights('my_model.h5')
        
        # evaluation metrics
        loss = model.evaluate(X_dev, Y_dev)
        print("Dev set loss = ", loss)
        
        # load model
        #model.load_weights('my_model.h5')
        
        #plot_model(model)
        
        # Prediction ----------------------------------------------------------                
        # train/dev set
        Y_nn_train = model.predict(X_train)
        Y_nn_dev   = model.predict(X_dev)
        
        # tiles
        Y_nn_A0  = model.predict(X_A0);  Y_nn_A180 = model.predict(X_A180)
        Y_nn_B0  = model.predict(X_B0);  Y_nn_B180 = model.predict(X_B180)
        
        Y_nn_A10 = model.predict(X_A10); Y_nn_A170 = model.predict(X_A170)
        Y_nn_B10 = model.predict(X_B10); Y_nn_B170 = model.predict(X_B170)
        
        Y_nn_A260 = model.predict(X_A260); Y_nn_A190 = model.predict(X_A190)
        Y_nn_B260 = model.predict(X_B260); Y_nn_B190 = model.predict(X_B190)
        
        Y_nn_A20 = model.predict(X_A20); Y_nn_A200 = model.predict(X_A200)
        
        Y_nn_B20 = model.predict(X_B20); Y_nn_B200 = model.predict(X_B200)
        
        # whole building
        Y_nn_highRise = model.predict(X_highRise)

        # evaluation metrics
        loss = model.evaluate(X_highRise, Y_nn_highRise)
        print("Test set loss = ", loss)
        
        # neural net tiles
        tile_nn_A = tileA(); 
        tile_nn_B = tileB(); 
                            
        # image to flat               
        y_nn_A0   = tile_nn_A.image_to_flat(Y_nn_A0)
        y_nn_A180 = tile_nn_A.image_to_flat(Y_nn_A180)
        y_nn_B0   = tile_nn_B.image_to_flat(Y_nn_B0)
        y_nn_B180 = tile_nn_B.image_to_flat(Y_nn_B180)   

        y_nn_A10  = tile_nn_A.image_to_flat(Y_nn_A10)
        y_nn_A170 = tile_nn_A.image_to_flat(Y_nn_A170)
        y_nn_B10  = tile_nn_B.image_to_flat(Y_nn_B10)
        y_nn_B170 = tile_nn_B.image_to_flat(Y_nn_B170)  

        y_nn_A260 = tile_nn_A.image_to_flat(Y_nn_A260)
        y_nn_A190 = tile_nn_A.image_to_flat(Y_nn_A190)
        y_nn_B260 = tile_nn_B.image_to_flat(Y_nn_B260)
        y_nn_B190 = tile_nn_B.image_to_flat(Y_nn_B190)  
        
        y_nn_A20  = tile_nn_A.image_to_flat(Y_nn_A20)
        y_nn_A200 = tile_nn_A.image_to_flat(Y_nn_A200)
        
        y_nn_B20  = tile_nn_B.image_to_flat(Y_nn_B20)
        y_nn_B200 = tile_nn_B.image_to_flat(Y_nn_B200)   
        
        y_nn_all = highRise.image_to_flat(Y_nn_highRise)
        
        # train/dev errors        
        MSE_A0   = error(tile_A0.rmsCp,   y_nn_A0)
        MSE_B0   = error(tile_B0.rmsCp,   y_nn_B0)
        MSE_A10  = error(tile_A10.rmsCp,  y_nn_A10)
        MSE_B10  = error(tile_B10.rmsCp,  y_nn_B10)
        MSE_A170 = error(tile_A170.rmsCp, y_nn_A170)
        MSE_B170 = error(tile_B170.rmsCp, y_nn_B170)
        MSE_A180 = error(tile_A180.rmsCp, y_nn_A180)
        MSE_B180 = error(tile_B180.rmsCp, y_nn_B180)
        MSE_A190 = error(tile_A190.rmsCp, y_nn_A190)
        MSE_B190 = error(tile_B190.rmsCp, y_nn_B190)
        MSE_A260 = error(tile_A260.rmsCp, y_nn_A260)
        MSE_B260 = error(tile_B260.rmsCp, y_nn_B260)
        
        MSE_train = np.mean([MSE_A0, MSE_B0, MSE_A180, MSE_B180, MSE_A10, MSE_B10,
                             MSE_A170, MSE_B170, MSE_A260, MSE_B190])
        print('Train error: ', MSE_train)
        
        MSE_dev = np.mean([MSE_B0, MSE_B180, MSE_A190, MSE_B260])
        print('Dev error: ', MSE_dev)
        
        # test errors                
        MSE_all = error(highRise.rmsCp, y_nn_all)
        print('Test error all: ', MSE_all)
        
        MSE_A20 = error(tile_A20.rmsCp, y_nn_A20)
        print('Test error A20: ', MSE_A20)
        
        MSE_B20 = error(tile_B20.rmsCp, y_nn_B20)
        print('Test error B20: ', MSE_B20)

        # contours ------------------------------------------------------------               
        '''
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        tile_A0.plot_RMSContour(y_nn_A0); tile_A180.plot_RMSContour(y_nn_A180)
        tile_B0.plot_RMSContour(y_nn_B0); tile_B180.plot_RMSContour(y_nn_B180)
        ax1.set_title(r'CNN: $C_{p''}$ at $0/180^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()
        
        plt.figure(figsize=(5,5))        
        ax1 = plt.subplot(1,1,1)
        tile_A0.plot_RMSContour(tile_A0.rmsCp); tile_A180.plot_RMSContour(tile_A180.rmsCp)
        tile_B0.plot_RMSContour(tile_B0.rmsCp); tile_B180.plot_RMSContour(tile_B180.rmsCp)
        ax1.set_title(r'Exp: $C_{p''}$ at $0/180^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()     
        '''
        
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        highRise.plot_RMSContour(y_nn_all)
        ax1.set_title(r'$C_{p,rms}$ at $0,180^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()
        #plt.savefig('/home/giacomol/Desktop/cprms_CNN_all.png')
        
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        highRise.plot_RMSContour(highRise.rmsCp)
        ax1.set_title(r'$C_{p,rms}$ at $0,180^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()
        #plt.savefig('/home/giacomol/Desktop/cprms_LES_all.png')          
        
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        tile_A20.plot_RMSContour(y_nn_A20);
        tile_B20.plot_RMSContour(y_nn_B20);
        ax1.set_title(r'$C_{p,rms}$ at $20^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()        
        #plt.savefig('/home/giacomol/Desktop/cprms_NN_20.png') 
        
        plt.figure(figsize=(5,5))        
        ax1 = plt.subplot(1,1,1)
        tile_A20.plot_RMSContour(tile_A20.rmsCp);
        tile_B20.plot_RMSContour(tile_B20.rmsCp);
        ax1.set_title(r'$C_{p.rms}$ at $20^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()   
        #plt.savefig('/home/giacomol/Desktop/cprms_exp_20.png')         
        
        '''
        plt.figure(figsize=(10,5))
        ax1 = plt.subplot(1,2,1)
        tile_A10.plot_RMSContour(y_nn_A10); tile_A170.plot_RMSContour(y_nn_A170)
        tile_B10.plot_RMSContour(y_nn_B10); tile_B170.plot_RMSContour(y_nn_B170)
        ax1.set_title(r'NN: $C_{p''}$ at $10-170^\circ$')
        plt.colorbar()
        
        ax1 = plt.subplot(1,2,2)
        tile_A10.plot_RMSContour(tile_A10.rmsCp); tile_A170.plot_RMSContour(tile_A170.rmsCp)
        tile_B10.plot_RMSContour(tile_B10.rmsCp); tile_B170.plot_RMSContour(tile_B170.rmsCp)
        ax1.set_title(r'Exp: $C_{p''}$ at $10-170^\circ$')
        plt.colorbar()

        plt.figure(figsize=(10,5))
        ax1 = plt.subplot(1,2,1)
        tile_A260.plot_RMSContour(y_nn_A260); tile_A190.plot_RMSContour(y_nn_A190)
        tile_B260.plot_RMSContour(y_nn_B260); tile_B190.plot_RMSContour(y_nn_B190)
        ax1.set_title(r'NN: $C_{p''}$ at $260-190^\circ$')
        plt.colorbar()
        
        ax1 = plt.subplot(1,2,2)
        tile_A260.plot_RMSContour(tile_A260.rmsCp); tile_A190.plot_RMSContour(tile_A190.rmsCp)
        tile_B260.plot_RMSContour(tile_B260.rmsCp); tile_B190.plot_RMSContour(tile_B190.rmsCp)
        ax1.set_title(r'Exp: $C_{p''}$ at $260-190^\circ$')
        plt.colorbar()
        
        plt.figure(figsize=(10,5))
        ax1 = plt.subplot(1,2,1)
        tile_A200.plot_RMSContour(y_nn_A200);
        tile_B200.plot_RMSContour(y_nn_B200);
        ax1.set_title(r'NN: $C_{p''}$ at $200^\circ$')
        plt.colorbar()
        
        ax1 = plt.subplot(1,2,2)
        tile_A200.plot_RMSContour(tile_A200.rmsCp);
        tile_B200.plot_RMSContour(tile_B200.rmsCp);
        ax1.set_title(r'Exp: $C_{p''}$ at $200^\circ$')
        plt.colorbar()
        '''
        '''
        plt.figure(figsize=(10,5))
        ax7 = plt.subplot(1,2,1)
        highRise.plot_RMSContour(y_nn[:,0,0]);
        ax7.set_title(r'NN RMS Estimate, $\theta = 0^\circ, 180^\circ$')
        plt.colorbar()
        
        ax8 = plt.subplot(1,2,2)
        highRise.plot_RMSContour(highRise.rmsCp);
        ax8.set_title(r'Experimental RMS Estimate, $\theta = 0^\circ, 180^\circ$')
        plt.colorbar()
        '''
        return loss
    
    return eval_model()

# Grid search configs ---------------------------------------------------------
# space of hyper-parameters
space = {'epochs' : hp.choice('epochs', [300, 500, 1000]),         
         'lr': hp.choice('lr', [0.001, 0.01, 0.1, 1]),
         'layers': hp.choice('layers', np.arange(1, 10, dtype=np.int32)),
         'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
         'dropout': hp.choice('dropout', [0.8, 1]),
         'batchNorm': hp.choice('batchNorm', [0, 1]),                             
         'filters': hp.choice('filters', np.arange(1, 20, dtype=np.int32)),
         'conv_size': hp.choice('conv_size', [1, 2, 3]),
         'pool_size': 1}

# train model -----------------------------------------------------------------
def trainer(hparams):
    
    # optimal model
    train(hparams)
    
    # model selection    
    #trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp1')    
    '''trials = Trials()
    
    best = fmin(train, space, algo=tpe.suggest, max_evals=500, trials=trials)
    opt_hparams = space_eval(space, best)
    print('optimal hyprparameters = ', opt_hparams)
    
    #train_hist = trials.train_history()
    
    return opt_hparams'''
