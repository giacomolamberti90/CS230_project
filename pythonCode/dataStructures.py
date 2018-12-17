import numpy as np
from util import *

# data structures -------------------------------------------------------------
Uref = 10.3
kref = 1.4

# tile A at 0deg       
tile_A0 = tileA()
tile_A0.angle  = 0
tile_A0.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A0.meanCp = np.genfromtxt('../RANS/cp_mean_A0.out')
tile_A0.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A0.out')
#tile_A0.rmsCp  = np.genfromtxt('../LES/cp_rms_A0.out')
tile_A0.k      = np.genfromtxt('../RANS/k_A0.out')/kref
tile_A0.U      = np.genfromtxt('../RANS/U_A0.out')/Uref
tile_A0.x      = np.genfromtxt('../RANS/features_A0.out')

# tile A at 10deg       
tile_A10 = tileA()
tile_A10.angle  = 10
tile_A10.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A10.meanCp = np.genfromtxt('../RANS/cp_mean_A10.out')
tile_A10.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A10.out')
tile_A10.k      = np.genfromtxt('../RANS/k_A10.out')/kref
tile_A10.U      = np.genfromtxt('../RANS/U_A10.out')/Uref
tile_A10.x      = np.genfromtxt('../RANS/features_A10.out')

# tile A at 20deg       
tile_A20 = tileA()
tile_A20.angle  = 20
tile_A20.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A20.meanCp = np.genfromtxt('../RANS/cp_mean_A20.out')
tile_A20.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A20.out')
tile_A20.k      = np.genfromtxt('../RANS/k_A20.out')/kref
tile_A20.U      = np.genfromtxt('../RANS/U_A20.out')/Uref
tile_A20.x      = np.genfromtxt('../RANS/features_A20.out')

# tile A at 170deg       
tile_A170 = tileA()
tile_A170.angle  = 170
tile_A170.coords = np.genfromtxt('../PoliMi/coords_A180')
tile_A170.meanCp = np.genfromtxt('../RANS/cp_mean_A170.out')
tile_A170.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A170.out')
tile_A170.k      = np.genfromtxt('../RANS/k_A170.out')/kref
tile_A170.U      = np.genfromtxt('../RANS/U_A170.out')/Uref
tile_A170.x      = np.genfromtxt('../RANS/features_A170.out')

# tile A at 180deg       
tile_A180 = tileA()
tile_A180.angle  = 180
tile_A180.coords = np.genfromtxt('../PoliMi/coords_A180')
tile_A180.meanCp = np.genfromtxt('../RANS/cp_mean_A180.out')
tile_A180.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A180.out')
#tile_A180.rmsCp  = np.genfromtxt('../LES/cp_rms_A180.out')
tile_A180.k      = np.genfromtxt('../RANS/k_A180.out')/kref
tile_A180.U      = np.genfromtxt('../RANS/U_A180.out')/Uref
tile_A180.x      = np.genfromtxt('../RANS/features_A180.out')

# tile A at 190deg       
tile_A190 = tileA()
tile_A190.angle  = 190
tile_A190.coords = np.genfromtxt('../PoliMi/coords_A180')
tile_A190.meanCp = np.genfromtxt('../RANS/cp_mean_A190.out')
tile_A190.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A190.out')
tile_A190.k      = np.genfromtxt('../RANS/k_A190.out')/kref
tile_A190.U      = np.genfromtxt('../RANS/U_A190.out')/Uref
tile_A190.x      = np.genfromtxt('../RANS/features_A190.out')

# tile A at 200deg       
tile_A200 = tileA()
tile_A200.angle  = 200
tile_A200.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A200.meanCp = np.genfromtxt('../RANS/cp_mean_A200.out')
tile_A200.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A200.out')
tile_A200.k      = np.genfromtxt('../RANS/k_A200.out')/kref
tile_A200.U      = np.genfromtxt('../RANS/U_A200.out')/Uref
tile_A200.x      = np.genfromtxt('../RANS/features_A200.out')

# tile A at 200deg       
tile_A260 = tileA()
tile_A260.angle  = 260
tile_A260.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A260.meanCp = np.genfromtxt('../RANS/cp_mean_A260.out')
tile_A260.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A260.out')
tile_A260.k      = np.genfromtxt('../RANS/k_A260.out')/kref
tile_A260.U      = np.genfromtxt('../RANS/U_A260.out')/Uref
tile_A260.x      = np.genfromtxt('../RANS/features_A260.out')

# tile B at 0deg       
tile_B0 = tileB()
tile_B0.angle  = 0
tile_B0.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B0.meanCp = np.genfromtxt('../RANS/cp_mean_B0.out')
tile_B0.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B0.out')
#tile_B0.rmsCp  = np.genfromtxt('../LES/cp_rms_B0.out')
tile_B0.k      = np.genfromtxt('../RANS/k_B0.out')/kref
tile_B0.U      = np.genfromtxt('../RANS/U_B0.out')/Uref
tile_B0.x      = np.genfromtxt('../RANS/features_B0.out')

# tile B at 10deg       
tile_B10 = tileB()
tile_B10.angle  = 10
tile_B10.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B10.meanCp = np.genfromtxt('../RANS/cp_mean_B10.out')
tile_B10.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B10.out')
tile_B10.k      = np.genfromtxt('../RANS/k_B10.out')/kref
tile_B10.U      = np.genfromtxt('../RANS/U_B10.out')/Uref
tile_B10.x      = np.genfromtxt('../RANS/features_B10.out')

# tile B at 20deg       
tile_B20 = tileB()
tile_B20.angle  = 20
tile_B20.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B20.meanCp = np.genfromtxt('../RANS/cp_mean_B20.out')
tile_B20.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B20.out')
tile_B20.k      = np.genfromtxt('../RANS/k_B20.out')/kref
tile_B20.U      = np.genfromtxt('../RANS/U_B20.out')/Uref
tile_B20.x      = np.genfromtxt('../RANS/features_B20.out')

# tile B at 170deg       
tile_B170 = tileB()
tile_B170.angle  = 170
tile_B170.coords = np.genfromtxt('../PoliMi/coords_B180')
tile_B170.meanCp = np.genfromtxt('../RANS/cp_mean_B170.out')
tile_B170.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B170.out')
tile_B170.k      = np.genfromtxt('../RANS/k_B170.out')/kref
tile_B170.U      = np.genfromtxt('../RANS/U_B170.out')/Uref
tile_B170.x      = np.genfromtxt('../RANS/features_B170.out')

# tile B at 180deg       
tile_B180 = tileB()
tile_B180.angle  = 180
tile_B180.coords = np.genfromtxt('../PoliMi/coords_B180')
tile_B180.meanCp = np.genfromtxt('../RANS/cp_mean_B180.out')
tile_B180.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B180.out')
#tile_B180.rmsCp  = np.genfromtxt('../LES/cp_rms_B180.out')
tile_B180.k      = np.genfromtxt('../RANS/k_B180.out')/kref
tile_B180.U      = np.genfromtxt('../RANS/U_B180.out')/Uref
tile_B180.x      = np.genfromtxt('../RANS/features_B180.out')

# tile B at 190deg       
tile_B190 = tileB()
tile_B190.angle  = 190
tile_B190.coords = np.genfromtxt('../PoliMi/coords_B180')
tile_B190.meanCp = np.genfromtxt('../RANS/cp_mean_B190.out')
tile_B190.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B190.out')
tile_B190.k      = np.genfromtxt('../RANS/k_B190.out')/kref
tile_B190.U      = np.genfromtxt('../RANS/U_B190.out')/Uref
tile_B190.x      = np.genfromtxt('../RANS/features_B190.out')

# tile B at 200deg       
tile_B200 = tileB()
tile_B200.angle  = 200
tile_B200.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B200.meanCp = np.genfromtxt('../RANS/cp_mean_B200.out')
tile_B200.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B200.out')
tile_B200.k      = np.genfromtxt('../RANS/k_B200.out')/kref
tile_B200.U      = np.genfromtxt('../RANS/U_B200.out')/Uref
tile_B200.x      = np.genfromtxt('../RANS/features_B200.out')

# tile B at 200deg       
tile_B260 = tileB()
tile_B260.angle  = 260
tile_B260.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B260.meanCp = np.genfromtxt('../RANS/cp_mean_B260.out')
tile_B260.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B260.out')
tile_B260.k      = np.genfromtxt('../RANS/k_B260.out')/kref
tile_B260.U      = np.genfromtxt('../RANS/U_B260.out')/Uref
tile_B260.x      = np.genfromtxt('../RANS/features_B260.out')
                   
# data frames
data_A0   = tile_A0.pressureDataFrame();   data_B0   = tile_B0.pressureDataFrame()
data_A10  = tile_A10.pressureDataFrame();  data_B10  = tile_B10.pressureDataFrame()
data_A20  = tile_A20.pressureDataFrame();  data_B20  = tile_B20.pressureDataFrame()
data_A170 = tile_A170.pressureDataFrame(); data_B170 = tile_B170.pressureDataFrame()
data_A180 = tile_A180.pressureDataFrame(); data_B180 = tile_B180.pressureDataFrame()
data_A190 = tile_A190.pressureDataFrame(); data_B190 = tile_B190.pressureDataFrame()
data_A200 = tile_A200.pressureDataFrame(); data_B200 = tile_B200.pressureDataFrame()
data_A260 = tile_A260.pressureDataFrame(); data_B260 = tile_B260.pressureDataFrame()

# 2D images
X_A0, Y_A0     = tile_A0.flat_to_image()
X_B0, Y_B0     = tile_B0.flat_to_image()
X_A180, Y_A180 = tile_A180.flat_to_image()
X_B180, Y_B180 = tile_B180.flat_to_image()

X_A10, Y_A10   = tile_A10.flat_to_image()
X_B10, Y_B10   = tile_B10.flat_to_image()
X_A170, Y_A170 = tile_A170.flat_to_image()
X_B170, Y_B170 = tile_B170.flat_to_image()

X_A260, Y_A260 = tile_A260.flat_to_image()
X_B260, Y_B260 = tile_B260.flat_to_image()
X_A190, Y_A190 = tile_A190.flat_to_image()
X_B190, Y_B190 = tile_B190.flat_to_image()

X_A20, Y_A20   = tile_A20.flat_to_image()
X_B20, Y_B20   = tile_B20.flat_to_image()

X_A200, Y_A200 = tile_A200.flat_to_image()
X_B200, Y_B200 = tile_B200.flat_to_image()

# whole building --------------------------------------------------------------
highRise = highRise()
highRise.angle  = 0
highRise.coords = np.genfromtxt('../LES/dataframe')[:,1:]
highRise.rmsCp  = np.genfromtxt('../LES/dataframe')[:,0]
highRise.meanCp = np.genfromtxt('../RANS/dataframe')[:,0]
highRise.k      = np.genfromtxt('../RANS/dataframe')[:,1]/kref
highRise.U      = np.genfromtxt('../RANS/dataframe')[:,2]/Uref
highRise.x      = np.genfromtxt('../RANS/dataframe')[:,3:]

data_highRise = highRise.pressureDataFrame()

# 2D images
X_highRise, Y_highRise = highRise.flat_to_image()