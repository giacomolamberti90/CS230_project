# Libraries 
import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm

class tile():
    
    def __init__(self):
        
        self.coords = [] # pressure taps coordinates
        self.taps   = [] # pressure taps names
        self.meanCp = [] # mean pressure coefficient
        self.rmsCp  = [] # rms pressure coeffcient
        self.k      = [] # turbulence kinetic energy
        self.U      = [] # mean velocity
        self.angle  = [] # wind direction
    
    def pressureDataFrame(self):
        
        # define data-frame
        data = np.column_stack((self.rmsCp, self.meanCp, self.k, self.U))
        dataFrame = pd.DataFrame(data, columns=['rmsCp', 'meanCp', 'k', 'U'])  
        
        return dataFrame
    
    def plot_RMSContour(self, y):
               
        # Delauney triangulation
        triang = tri.Triangulation(self.coords[:,0], self.coords[:,1])
        levels = np.arange(0, 0.5, 0.005)
        
        # plot rms contour
        plt.rcParams.update({'font.size': 16})
        plt.tricontourf(triang, np.reshape(y, (len(y),)), levels=levels, cmap='hot_r')
        plt.plot(self.coords[:,0], self.coords[:,1], '.k', markersize=1)
        plt.xlabel(r"$x[m]$"); plt.ylabel(r"$y[m]$")
        plt.tight_layout()
        
                
class tileA(tile):

    def __init__(self):        
        super().__init__()
                
        # read pressure taps
        self.taps = np.genfromtxt('../PoliMi/taps_A')[:,1:3]
                    
    def plot_RMSProfiles(self, y):
                    
        # indices tile A
        ind_A = []
        for i in range(0,224):	
            ind_A.append(int(str(int(self.taps[i,0])) + str(int(self.taps[i,1]))))
        
        # plot rms profiles on tile A
        for k in range(1,15):
    	
            ind = [i for i, x in enumerate(ind_A) if x == k]
                	
            plt.subplot(3,5,k)
            plt.plot(self.coords[ind,0], self.rmsCp[ind], '.r') 
            plt.plot(self.coords[ind,0], y[ind], '.k')
            plt.xlim([0, 1]); plt.ylim([0, 0.5])
            plt.xlabel(r"$x[m]$"); plt.ylabel(r"$C_{p'}$")
            plt.tight_layout()

class tileB(tile):
    
    def __init__(self):    
        super().__init__()

        # read pressure taps
        self.taps = np.genfromtxt('../PoliMi/taps_B')[:,3:]
        
    def plot_RMSProfiles(self, y):

        # indices tile B
        ind_B = []
        for i in range(0,223):	
            ind_B.append(int(str(int(self.taps[i,0])) + str(int(self.taps[i,1]))))
                
        # plot rms profiles on tile B        
        for k in range(1,20):
	
            ind = [i for i, x in enumerate(ind_B) if x == k]
	
            plt.subplot(4,5,k)
            plt.plot(self.coords[ind,0], self.rmsCp[ind], '.r') 
            plt.plot(self.coords[ind,0], y[ind], '.k')
            plt.xlim([0, 1]); plt.ylim([0, 0.5])	
            plt.xlabel(r"$x[m]$"); plt.ylabel(r"$C_{p'}$")
            plt.tight_layout()

def plot(y_exp, y_pred):
    
    plt.figure()
    plt.plot(y_exp, y_pred, '.r')
    plt.plot(np.linspace(0,1), np.linspace(0,1), 'k')
    plt.plot(np.linspace(0,1), 0.9*np.linspace(0,1), '--k')
    plt.plot(np.linspace(0,1), 1.1*np.linspace(0,1), '--k')
    plt.xlabel(r"$C_{p',exp}$"); plt.ylabel(r"$C_{p',nn}$")
    plt.xlim([0,0.5]); plt.ylim([0,0.5])

# data structures

# tile A at 0deg       
tile_A0 = tileA()
tile_A0.angle  = 0
tile_A0.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A0.meanCp = np.genfromtxt('../RANS/cp_mean_A0.out')
tile_A0.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A0.out')
tile_A0.k      = np.genfromtxt('../RANS/k_A0.out')
tile_A0.U      = np.genfromtxt('../RANS/U_A0.out')

# tile A at 180deg       
tile_A180 = tileA()
tile_A180.angle  = 180
tile_A180.coords = np.genfromtxt('../PoliMi/coords_A180')
tile_A180.meanCp = np.genfromtxt('../RANS/cp_mean_A180.out')
tile_A180.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A180.out')
tile_A180.k      = np.genfromtxt('../RANS/k_A180.out')
tile_A180.U      = np.genfromtxt('../RANS/U_A180.out')

# tile A at 20deg       
tile_A20 = tileA()
tile_A20.angle  = 20
tile_A20.coords = np.genfromtxt('../PoliMi/coords_A0')
tile_A20.meanCp = np.genfromtxt('../RANS/cp_mean_A20.out')
tile_A20.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_A20.out')
tile_A20.k      = np.genfromtxt('../RANS/k_A20.out')
tile_A20.U      = np.genfromtxt('../RANS/U_A20.out')

# tile B at 0deg       
tile_B0 = tileB()
tile_B0.angle  = 0
tile_B0.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B0.meanCp = np.genfromtxt('../RANS/cp_mean_B0.out')
tile_B0.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B0.out')
tile_B0.k      = np.genfromtxt('../RANS/k_B0.out')
tile_B0.U      = np.genfromtxt('../RANS/U_B0.out')

# tile B at 180deg       
tile_B180 = tileB()
tile_B180.angle  = 180
tile_B180.coords = np.genfromtxt('../PoliMi/coords_B180')
tile_B180.meanCp = np.genfromtxt('../RANS/cp_mean_B180.out')
tile_B180.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B180.out')
tile_B180.k      = np.genfromtxt('../RANS/k_B180.out')
tile_B180.U      = np.genfromtxt('../RANS/U_B180.out')

# tile B at 20deg       
tile_B20 = tileB()
tile_B20.angle  = 20
tile_B20.coords = np.genfromtxt('../PoliMi/coords_B0')
tile_B20.meanCp = np.genfromtxt('../RANS/cp_mean_B20.out')
tile_B20.rmsCp  = np.genfromtxt('../PoliMi/cp_rms_exp_B20.out')
tile_B20.k      = np.genfromtxt('../RANS/k_B20.out')
tile_B20.U      = np.genfromtxt('../RANS/U_B20.out')
