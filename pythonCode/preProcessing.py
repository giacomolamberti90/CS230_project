from util import *
from dataStructures import *

# Data --------------------------------------------------------------------    
# full data frame
data_A0   = tile_A0.pressureDataFrame();   data_B0   = tile_B0.pressureDataFrame()
data_A10  = tile_A10.pressureDataFrame();  data_B10  = tile_B10.pressureDataFrame()
data_A20  = tile_A20.pressureDataFrame();  data_B20  = tile_B20.pressureDataFrame()
data_A180 = tile_A180.pressureDataFrame(); data_B180 = tile_B180.pressureDataFrame()
data_A190 = tile_A190.pressureDataFrame(); data_B190 = tile_B190.pressureDataFrame()
data_A200 = tile_A200.pressureDataFrame(); data_B200 = tile_B200.pressureDataFrame()

data_highRise = highRise.pressureDataFrame()

# plot input data -------------------------------------------------------------

i = 0

levels = np.arange(-1, 1, 0.05)

# 0-180 -------------------------------
'''
plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
highRise.plot_RMSContour(highRise.rmsCp)
plt.colorbar()

ax1 = plt.subplot(1,2,2)
highRise.plot_Contour(highRise.x[:,i], levels)
plt.colorbar()

'''
plt.figure(figsize=(18,10))
ax1 = plt.subplot(2,4,1)
levels = np.arange(-1.5, 0, 0.05)
tile_A0.plot_Contour(tile_A0.meanCp, levels); tile_A180.plot_Contour(tile_A180.meanCp, levels)
tile_B0.plot_Contour(tile_B0.meanCp, levels); tile_B180.plot_Contour(tile_B180.meanCp, levels)
ax1.set_title(r'$x_1$')
plt.colorbar()

ax1 = plt.subplot(2,4,2)
levels = np.arange(0, 4, 0.05)
tile_A0.plot_Contour(tile_A0.k, levels); tile_A180.plot_Contour(tile_A180.k, levels)
tile_B0.plot_Contour(tile_B0.k, levels); tile_B180.plot_Contour(tile_B180.k, levels)
ax1.set_title(r'$x_2$')
plt.colorbar()

ax1 = plt.subplot(2,4,3)
levels = np.arange(0.5, 1.5, 0.05)
tile_A0.plot_Contour(tile_A0.U, levels); tile_A180.plot_Contour(tile_A180.U, levels)
tile_B0.plot_Contour(tile_B0.U, levels); tile_B180.plot_Contour(tile_B180.U, levels)
ax1.set_title(r'$x_3$')
plt.colorbar()

ax1 = plt.subplot(2,4,4)
levels = np.arange(-0.4, 0.4, 0.05)
tile_A0.plot_Contour(tile_A0.x[:,0], levels); tile_A180.plot_Contour(tile_A180.x[:,0], levels)
tile_B0.plot_Contour(tile_B0.x[:,0], levels); tile_B180.plot_Contour(tile_B180.x[:,0], levels)
ax1.set_title(r'$x_4$')
plt.colorbar()

ax1 = plt.subplot(2,4,5)
levels = np.arange(-1, 1, 0.05)
tile_A0.plot_Contour(tile_A0.x[:,1], levels); tile_A180.plot_Contour(tile_A180.x[:,1], levels)
tile_B0.plot_Contour(tile_B0.x[:,1], levels); tile_B180.plot_Contour(tile_B180.x[:,1], levels)
ax1.set_title(r'$x_5$')
plt.colorbar()

ax1 = plt.subplot(2,4,6)
levels = np.arange(0, 0.5, 0.05)
tile_A0.plot_Contour(tile_A0.x[:,2], levels); tile_A180.plot_Contour(tile_A180.x[:,2], levels)
tile_B0.plot_Contour(tile_B0.x[:,2], levels); tile_B180.plot_Contour(tile_B180.x[:,2], levels)
ax1.set_title(r'$x_6$')
plt.colorbar()

ax1 = plt.subplot(2,4,7)
levels = np.arange(-1, 1, 0.05)
tile_A0.plot_Contour(tile_A0.x[:,3], levels); tile_A180.plot_Contour(tile_A180.x[:,3], levels)
tile_B0.plot_Contour(tile_B0.x[:,3], levels); tile_B180.plot_Contour(tile_B180.x[:,3], levels)
ax1.set_title(r'$x_7$')
plt.colorbar()

ax1 = plt.subplot(2,4,8)
levels = np.arange(0.2, 0.8, 0.05)
tile_A0.plot_Contour(tile_A0.x[:,4], levels); tile_A180.plot_Contour(tile_A180.x[:,4], levels)
tile_B0.plot_Contour(tile_B0.x[:,4], levels); tile_B180.plot_Contour(tile_B180.x[:,4], levels)
ax1.set_title(r'$x_8$')
plt.colorbar()

'''
# 10-170 -------------------------------
plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
tile_A10.plot_RMSContour(tile_A10.rmsCp); tile_A170.plot_RMSContour(tile_A170.rmsCp)
tile_B10.plot_RMSContour(tile_B10.rmsCp); tile_B170.plot_RMSContour(tile_B170.rmsCp)
plt.colorbar()

ax2 = plt.subplot(1,2,2)
tile_A10.plot_Contour(tile_A10.x[:,i], levels); tile_A170.plot_Contour(tile_A170.x[:,i], levels)
tile_B10.plot_Contour(tile_B10.x[:,i], levels); tile_B170.plot_Contour(tile_B170.x[:,i], levels)
plt.colorbar()

# 190-260 -------------------------------
plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,2,1)
tile_A260.plot_RMSContour(tile_A260.rmsCp); tile_A190.plot_RMSContour(tile_A190.rmsCp)
tile_B260.plot_RMSContour(tile_B260.rmsCp); tile_B190.plot_RMSContour(tile_B190.rmsCp)
plt.colorbar()

ax2 = plt.subplot(1,2,2)
tile_A260.plot_Contour(tile_A260.x[:,i], levels); tile_A190.plot_Contour(tile_A190.x[:,i], levels)
tile_B260.plot_Contour(tile_B260.x[:,i], levels); tile_B190.plot_Contour(tile_B190.x[:,i], levels)
plt.colorbar()
'''