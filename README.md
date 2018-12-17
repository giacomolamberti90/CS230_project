# CS230_project

# PoliMi 

directory that contains the experimental data:

	- coordinates of the pressure taps
	
	- rms pressure coefficient
	
	- pressure taps names
	
A0 means tile A at 0 deg

A180 means tile A at 180 deg...

# RANS

directory that contains the RANS simulations data
	
# LES

directory that contains the LES simulations data (high-fidelity simulations), used only at test time.

# pythonCode

directory containing the code:

	- main.py: call trainer
	
	- trainer.py: fit the artificial neural network and perform hyperparameter search
	
	- trainer_conv2D.py: fit the convolutional neural network and perform hyperparameter search
	
	- dataStructures.py: define data-structures
	
	- util.py: define functions to read and post-process data
