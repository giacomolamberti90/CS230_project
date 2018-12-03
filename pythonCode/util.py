# Libraries 
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
        self.x      = [] # additional features from Ling etal 2015
    
    def pressureDataFrame(self):

        # define data-frame
        data = np.column_stack((self.rmsCp, self.meanCp, self.k, self.U, self.x))
        dataFrame = pd.DataFrame(data, columns=['rmsCp', 'meanCp', 'k', 'U', 'x1','x5', 'x6', 'x7', 'x11'])  
        
        return dataFrame
    
    def plot_RMSContour(self, y):
               
        # Delauney triangulation
        triang = tri.Triangulation(self.coords[:,0], self.coords[:,1])
        levels = np.arange(0, 0.5, 0.005)
        
        # plot rms contour
        plt.rcParams.update({'font.size': 16})
        plt.tricontourf(triang, np.reshape(y, (len(y),)), levels=levels, cmap='hot_r')
        plt.plot(self.coords[:,0], self.coords[:,1], '.k', markersize=1)
        plt.xlabel(r"$x[m]$"); plt.ylabel(r"$y[m]$"); plt.xlim(0,1); plt.ylim(0,2)
        plt.tight_layout() 
                    
    def plot_Contour(self, y, levels):
               
        # Delauney triangulation
        triang = tri.Triangulation(self.coords[:,0], self.coords[:,1])
        
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
        for i in range(224):	
            ind_A.append(int(str(int(self.taps[i,0])) + str(int(self.taps[i,1]))))
        
        # plot rms profiles on tile A
        for k in range(15):
    	
            ind = [i for i, x in enumerate(ind_A) if x == k]
                	
            plt.subplot(3,5,k)
            plt.plot(self.coords[ind,0], self.rmsCp[ind], '.r') 
            plt.plot(self.coords[ind,0], y[ind], '.k')
            plt.xlim([0, 1]); plt.ylim([0, 0.5])
            plt.xlabel(r"$x[m]$"); plt.ylabel(r"$C_{p'}$")
            plt.tight_layout()
            
    def flat_to_image(self):
                
        # read pressure taps
        taps = np.genfromtxt('../PoliMi/taps_A')[:,1:]
        
        # initialize image
        input_image  = np.zeros((1,19,15,8))
        output_image = np.zeros((1,19,15,1))
        
        # indices tile A
        for i in range(224):
            row = int(str(int(taps[i,0])) + str(int(taps[i,1])))
            col = int(str(int(taps[i,2])) + str(int(taps[i,3])))
            
            # features
            input_image[0, row-1, col-1, 0] = self.meanCp[i]
            input_image[0, row-1, col-1, 1] = self.k[i]/1.4
            input_image[0, row-1, col-1, 2] = self.U[i]/10.3
            input_image[0, row-1, col-1, 3] = self.x[i,0]
            input_image[0, row-1, col-1, 4] = self.x[i,1]
            input_image[0, row-1, col-1, 5] = self.x[i,2]
            input_image[0, row-1, col-1, 6] = self.x[i,3]
            input_image[0, row-1, col-1, 7] = self.x[i,4]
            
            # labels
            output_image[0, row-1, col-1, 0] = self.rmsCp[i] 
                        
        return input_image, output_image

    def image_to_flat(self, output_image):
        
        # intialize
        output_flat = np.zeros((224,1))
        
        # read pressure taps
        taps = np.genfromtxt('../PoliMi/taps_A')[:,1:]
        
        # indices tile A
        for i in range(224):
            row = int(str(int(taps[i,0])) + str(int(taps[i,1])))
            col = int(str(int(taps[i,2])) + str(int(taps[i,3])))
            
            # labels
            output_flat[i] = output_image[0, row-1, col-1, 0]
                        
        return output_flat
            

class tileB(tile):
    
    def __init__(self):    
        super().__init__()

        # read pressure taps
        self.taps = np.genfromtxt('../PoliMi/taps_B')[:,3:]
        
    def plot_RMSProfiles(self, y):

        # indices tile B
        ind_B = []
        for i in range(223):	
            ind_B.append(int(str(int(self.taps[i,0])) + str(int(self.taps[i,1]))))
                
        # plot rms profiles on tile B        
        for k in range(20):
	
            ind = [i for i, x in enumerate(ind_B) if x == k]
	
            plt.subplot(4,5,k)
            plt.plot(self.coords[ind,0], self.rmsCp[ind], '.r') 
            plt.plot(self.coords[ind,0], y[ind], '.k')
            plt.xlim([0, 1]); plt.ylim([0, 0.5])	
            plt.xlabel(r"$x[m]$"); plt.ylabel(r"$C_{p'}$")
            plt.tight_layout()
            
            
    def flat_to_image(self):
                
        # read pressure taps
        taps = np.genfromtxt('../PoliMi/taps_B')[:,1:]
        
        # initialize image
        input_image  = np.zeros((1,19,15,8))
        output_image = np.zeros((1,19,15,1))
        
        # indices tile A
        for i in range(223):	
            col = int(str(int(taps[i,0])) + str(int(taps[i,1])))
            row = int(str(int(taps[i,2])) + str(int(taps[i,3])))
            
            # features
            input_image[0, row-1, col-1, 0] = self.meanCp[i]
            input_image[0, row-1, col-1, 1] = self.k[i]/1.4
            input_image[0, row-1, col-1, 2] = self.U[i]/10.3
            input_image[0, row-1, col-1, 3] = self.x[i,0]
            input_image[0, row-1, col-1, 4] = self.x[i,1]
            input_image[0, row-1, col-1, 5] = self.x[i,2]
            input_image[0, row-1, col-1, 6] = self.x[i,3]
            input_image[0, row-1, col-1, 7] = self.x[i,4]
            
            # labels
            output_image[0, row-1, col-1, 0] = self.rmsCp[i] 
                        
        return input_image, output_image
    
    def image_to_flat(self, output_image):
        
        # intialize
        output_flat = np.zeros((223,1))
                
        # read pressure taps
        taps = np.genfromtxt('../PoliMi/taps_B')[:,1:]
        
        # indices tile A
        for i in range(223):
            col = int(str(int(taps[i,0])) + str(int(taps[i,1])))
            row = int(str(int(taps[i,2])) + str(int(taps[i,3])))
            
            # labels
            output_flat[i] = output_image[0, row-1, col-1, 0]
                        
        return output_flat

class highRise(tile):

    def __init__(self):        
        super().__init__()
            
    def flat_to_image(self):
                
        # initialize image
        input_image  = np.zeros((1,40,20,8))
        output_image = np.zeros((1,40,20,1))
        
        # indices all
        for i in range(40):
            for j in range(20):
                
                # features
                input_image[0,i,j,0] = self.meanCp[i*20+j]
                input_image[0,i,j,1] = self.k[i*20+j]/1.4
                input_image[0,i,j,2] = self.U[i*20+j]/10.3
                input_image[0,i,j,3] = self.x[i*20+j,0]
                input_image[0,i,j,4] = self.x[i*20+j,1]
                input_image[0,i,j,5] = self.x[i*20+j,2]
                input_image[0,i,j,6] = self.x[i*20+j,3]
                input_image[0,i,j,7] = self.x[i*20+j,4]
                
                # labels
                output_image[0,i,j,0] = self.rmsCp[i*20+j] 
                        
        return input_image, output_image

    def image_to_flat(self, output_image):
        
        # intialize
        output_flat = np.zeros((800,1))

        # indices all
        for i in range(40):
            for j in range(20):
                                
                # labels
                output_flat[i*20+j] = output_image[0,i,j,0]
                        
        return output_flat
    
def plot(y_exp, y_pred):
    
    plt.figure()
    plt.plot(y_exp, y_pred, '.r')
    plt.plot(np.linspace(0,1), np.linspace(0,1), 'k')
    plt.plot(np.linspace(0,1), 0.9*np.linspace(0,1), '--k')
    plt.plot(np.linspace(0,1), 1.1*np.linspace(0,1), '--k')
    plt.xlabel(r"$C_{p',exp}$"); plt.ylabel(r"$C_{p',nn}$")
    plt.xlim([0,0.5]); plt.ylim([0,0.5])
