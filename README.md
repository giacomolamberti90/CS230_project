# CS230_project

# PoliMi 
directory that contains the experimental data:
	- coordinates of the pressure taps
	- rms pressure coefficient
	- pressure taps names
	
A0 means tile A at 0 deg
A180 means tile A at 180 deg...

# RANS
directory that contains the RANS simulations data:
	- mean pressure coefficient (local)
	- turbulence kinetic energy (local)
	- mean velocity at inlet (just function of height)
	
# LES
directory that contains the LES simulations data (high-fidelity simulations), 
not used at this moement.

# pythonCode
directory containing the code:
	- main.py: fit the neural network and make plots
	- util.py: functions to read and post-process data
