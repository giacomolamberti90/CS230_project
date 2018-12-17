import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import keras.backend as K

from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv1D
from keras.models import Model
from keras.optimizers import Adam

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
    '''
    # train on random samples from all the tiles at 0-180 + 20-200
    data_train = data_A0.append([data_A20, data_A180, data_A200, 
                        data_B0, data_B20, data_B180, data_B200], ignore_index=True)

    # training set
    train   = data_train.sample(frac=0.8)
    x_train = np.ndarray((len(train), n)); y_train = np.ndarray((len(train),1))
    
    x_train[:,0] = train['meanCp']; x_train[:,1] = train['k'];  x_train[:,2] = train['U']
    x_train[:,3] = train['x1'];     x_train[:,4] = train['x5']; x_train[:,5] = train['x6']
    x_train[:,6] = train['x7'];     x_train[:,7] = train['x11']
    y_train[:,0] = train['rmsCp']
    
    # dev set
    dev   = data_train.drop(train.index, axis=0)
    x_dev = np.ndarray((len(dev), n)); y_dev = np.ndarray((len(dev), 1))
    
    x_dev[:,0] = dev['meanCp']; x_dev[:,1] = dev['k'];  x_dev[:,2] = dev['U']      
    x_dev[:,3] = dev['x1'];     x_dev[:,4] = dev['x5']; x_dev[:,5] = dev['x6']
    x_dev[:,6] = dev['x7'];     x_dev[:,7] = dev['x11']    
    y_dev[:,0] = dev['rmsCp']    
    '''
    # train on tiles at 0-180
    data_train = data_A0.append([data_A180, data_B0, data_B180, data_A10, data_A170, 
                                 data_B10, data_B170, data_A260, data_B190], ignore_index=True)
    
    data_dev   = data_B0.append([data_B180, data_A190, data_B260], ignore_index=True)
    
    # training set
    x_train = np.ndarray((data_train.shape[0], n)); y_train = np.ndarray((data_train.shape[0], 1))
    
    x_train[:,0] = data_train['meanCp']; x_train[:,1] = data_train['k'];  x_train[:,2] = data_train['U']
    x_train[:,3] = data_train['x5'];     x_train[:,4] = data_train['x6']; x_train[:,5] = data_train['x7']
    x_train[:,6] = data_train['x11'];    #x_train[:,7] = data_train['angle']
    y_train[:,0] = data_train['rmsCp']
    
    # dev set
    x_dev = np.ndarray((data_dev.shape[0], n)); y_dev = np.ndarray((data_dev.shape[0], 1))
    
    x_dev[:,0] = data_dev['meanCp']; x_dev[:,1] = data_dev['k'];  x_dev[:,2] = data_dev['U']      
    x_dev[:,3] = data_dev['x5'];     x_dev[:,4] = data_dev['x6']; x_dev[:,5] = data_dev['x7']
    x_dev[:,6] = data_dev['x11'];    #x_dev[:,7] = data_dev['angle']    
    y_dev[:,0] = data_dev['rmsCp']
    
    # indices of features
    ix = np.arange(1,n+1)
    
    # Model -------------------------------------------------------------------
    def model(input_shape):
    
        X_input = Input(shape=input_shape)
        
        # first layer
        X = Dense(hparams['units'], activation=hparams['activation'])(X_input)
        
        # hidden layers
        for i in range(hparams['layers']):
            X = Dense(hparams['units'], activation=hparams['activation'])(X)
            X = Dropout(rate=hparams['dropout'])(X)
            X = BatchNormalization()(X)
            
        # output layer
        X = Dense(1, activation='sigmoid')(X)
        
        # model instance
        model = Model(inputs=X_input, outputs=X)
        
        return model
        
    # Model instance ----------------------------------------------------------
    keras.backend.clear_session()
    tf.reset_default_graph()
    model = model(input_shape=(n,))
    
    # Evaluate model ----------------------------------------------------------
    def eval_model():
        
        ## optimization
        opt = Adam(lr=hparams['lr'])
        model.compile(loss='mse', optimizer=opt)
        
        # train
        np.random.seed(1)
        model.fit(x_train, y_train, epochs=hparams['epochs'])
        
        # save model
        model.save_weights('my_model.h5')
        
        # evaluation metrics
        loss = model.evaluate(x_dev, y_dev)
        print("Dev set loss = ", loss)
        
        # load model
        #model.load_weights('my_model.h5')
        
        # Prediction ----------------------------------------------------------
        y_nn_train = model.predict(x_train)
        y_nn_dev   = model.predict(x_dev)
        
        # prediction on pressure tiles
        y_nn_A0   = model.predict(data_A0.values[:,ix]);   y_nn_B0   = model.predict(data_B0.values[:,ix])
        y_nn_A10  = model.predict(data_A10.values[:,ix]);  y_nn_B10  = model.predict(data_B10.values[:,ix])
        y_nn_A20  = model.predict(data_A20.values[:,ix]);  y_nn_B20  = model.predict(data_B20.values[:,ix])
        y_nn_A170 = model.predict(data_A170.values[:,ix]); y_nn_B170 = model.predict(data_B170.values[:,ix])        
        y_nn_A180 = model.predict(data_A180.values[:,ix]); y_nn_B180 = model.predict(data_B180.values[:,ix])
        y_nn_A190 = model.predict(data_A190.values[:,ix]); y_nn_B190 = model.predict(data_B190.values[:,ix])
        y_nn_A200 = model.predict(data_A200.values[:,ix]); y_nn_B200 = model.predict(data_B200.values[:,ix])
        y_nn_A260 = model.predict(data_A260.values[:,ix]); y_nn_B260 = model.predict(data_B260.values[:,ix])
        
        # prediction on whole building
        y_nn_all = model.predict(data_highRise.values[:,1:n+1]);
                                        
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
        #plt.savefig('/home/giacomol/Desktop/cprms_ANN_all.png')
        
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        highRise.plot_RMSContour(highRise.rmsCp)
        ax1.set_title(r'$C_{p,rms}$ at $0,180^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()           
        
        plt.figure(figsize=(5,5))
        ax1 = plt.subplot(1,1,1)
        tile_A20.plot_RMSContour(y_nn_A20);
        tile_B20.plot_RMSContour(y_nn_B20);
        ax1.set_title(r'$C_{p,rms}$ at $20^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()   
        #plt.savefig('/home/giacomol/Desktop/cprms_ANN_20.png')
        
        plt.figure(figsize=(5,5))        
        ax1 = plt.subplot(1,1,1)
        tile_A20.plot_RMSContour(tile_A20.rmsCp);
        tile_B20.plot_RMSContour(tile_B20.rmsCp);
        ax1.set_title(r'$C_{p.rms}$ at $20^\circ$')
        cb = plt.colorbar(); cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()           
        
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
        # profiles ------------------------------------------------------------
        plt.figure(figsize=(15,8))
        tile_A0.plot_RMSProfiles(y_nn_A0)
        tile_A180.plot_RMSProfiles(y_nn_A180)
        
        plt.figure(figsize=(15,8))
        tile_B0.plot_RMSProfiles(y_nn_B0)
        tile_B180.plot_RMSProfiles(y_nn_B180)
        plt.show()
        
        plt.figure(figsize=(15,8))
        tile_A10.plot_RMSProfiles(y_nn_A10)
        tile_A190.plot_RMSProfiles(y_nn_A190)
        
        plt.figure(figsize=(15,8))
        tile_B10.plot_RMSProfiles(y_nn_B10)
        tile_B190.plot_RMSProfiles(y_nn_B190)
        plt.show()
        
        plt.figure(figsize=(15,8))
        tile_A20.plot_RMSProfiles(y_nn_A20)
        tile_A200.plot_RMSProfiles(y_nn_A200)
        
        plt.figure(figsize=(15,8))
        tile_B20.plot_RMSProfiles(y_nn_B20)
        tile_B200.plot_RMSProfiles(y_nn_B200)
        plt.show()
        '''
        return loss
    
    return eval_model()

# Grid search configs ---------------------------------------------------------
# space of hyper-parameters
space = {'epochs' : hp.choice('epochs', [100, 200, 300]),         
         'lr': hp.choice('lr', [0.001, 0.01, 0.1, 1]),
         'layers': hp.choice('layers', np.arange(1, 20, dtype=np.int32)),
         'units': hp.choice('units', np.arange(1, 10, dtype=np.int32)),
         'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
         'dropout': hp.choice('dropout', [0.6, 0.8, 1])}

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
    
