import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb 

from util import *

# full data frame
data_A0   = tile_A0.pressureDataFrame()
data_A180 = tile_A180.pressureDataFrame()
data_A20  = tile_A20.pressureDataFrame()

data_B0   = tile_B0.pressureDataFrame()
data_B180 = tile_B180.pressureDataFrame()
data_B20  = tile_B20.pressureDataFrame()

data = data_A0.append([data_A180, data_B0, data_B180], ignore_index=True)

# Data ------------------------------------------------------------------------
np.random.seed(1)

# training data
train   = data.sample(frac=0.8)
x_train = np.ndarray((len(train),3)); y_train = np.ndarray((len(train),1))

x_train[:,0] = train['meanCp']; x_train[:,1] = train['k']
x_train[:,2] = train['U'];      y_train[:,0] = train['rmsCp']

# test data
test   = data.drop(train.index, axis=0)
x_test = np.ndarray((len(test), 3)); y_test  = np.ndarray((len(test), 1))

x_test[:,0] = test['meanCp']; x_test[:,1] = test['k']
x_test[:,2] = test['U'];      y_test[:,0] = test['rmsCp']

# Feed-forward neural nerwotk -------------------------------------------------
model = tf.keras.Sequential()

# layers
np.random.seed(1)
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(units=3, activation='tanh', input_shape=x_train.shape))
#model.add(tf.keras.layers.Dense(units=1, activation='tanh'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=10, activation = 'relu', input_shape = x_train.shape))
model.add(tf.keras.layers.Dense(units=5, activation = 'relu'))
model.add(tf.keras.layers.Dense(units=3, activation = 'relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## loss
#model.compile(optimizer='rmsprop', loss='mse')
model.compile(optimizer='adam', loss='mse')

# train
np.random.seed(1)
#model.fit(x_train, y_train, epochs=100)
model.fit(x_train, y_train, epochs=500)

loss_and_metrics = model.evaluate(x_test, y_test)

# predict ---------------------------------------------------------------------
y_nn_train = model.predict(x_train)
y_nn_test  = model.predict(x_test)

# prediction on pressure tiles
y_nn_A0   = model.predict(data_A0.values[:,1:])
y_nn_A180 = model.predict(data_A180.values[:,1:])
y_nn_A20  = model.predict(data_A20.values[:,1:])

y_nn_B0   = model.predict(data_B0.values[:,1:])
y_nn_B180 = model.predict(data_B180.values[:,1:])
y_nn_B20  = model.predict(data_B20.values[:,1:])

# figures ---------------------------------------------------------------------
plot(y_train, y_nn_train)
plot(y_test,  y_nn_test)

plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
tile_A0.plot_RMSContour(y_nn_A0); tile_A180.plot_RMSContour(y_nn_A180)
tile_B0.plot_RMSContour(y_nn_B0); tile_B180.plot_RMSContour(y_nn_B180)
ax1.set_title(r'NN RMS Estimate, $\theta = 0^\circ, 180^\circ$')
plt.colorbar()

ax2 = plt.subplot(1,2,2)
tile_A0.plot_RMSContour(tile_A0.rmsCp); tile_A180.plot_RMSContour(tile_A180.rmsCp)
tile_B0.plot_RMSContour(tile_B0.rmsCp); tile_B180.plot_RMSContour(tile_B180.rmsCp)
ax2.set_title(r'Experimental RMS Estimate, $\theta = 0^\circ, 180^\circ$')
plt.colorbar()

plt.figure(figsize=(15,8))
tile_A0.plot_RMSProfiles(y_nn_A0)
tile_A180.plot_RMSProfiles(y_nn_A180)

plt.figure(figsize=(15,8))
tile_B0.plot_RMSProfiles(y_nn_B0)
tile_B180.plot_RMSProfiles(y_nn_B180)


plt.figure(figsize=(10,5))
ax3=plt.subplot(1,2,1)
tile_A20.plot_RMSContour(y_nn_A20);
tile_B20.plot_RMSContour(y_nn_B0); 
ax3.set_title(r'NN RMS Estimate, $\theta = 20^\circ$')
plt.colorbar()

ax4 = plt.subplot(1,2,2)
tile_A20.plot_RMSContour(tile_A20.rmsCp);
tile_B20.plot_RMSContour(tile_B20.rmsCp); 
ax4.set_title(r'Experimental RMS Estimate, $\theta = 20^\circ$')
plt.colorbar()

plt.show()

