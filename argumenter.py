#import numpy as np
from konstanter import AU, Msun

phi = 0.05 # angle
Tstar = 4500 # T Tauri surface temp [k]
M_centralstar = 0.5 * Msun # mass of central star [kg]
R = 3*(6.95700*10**8) # radius of T Tauri star [m]
#L = 4*np.pi*(R**2)*k*(Tstar**4)

nr = 100.
r_in = 0.03 * AU
r_yd = 100. * AU

dlnHpdlr = 9/7
alpha = 10**-3 # turbulence-parameter
q = 0.5 # turbulence-parameter
delta = 0.8 

rho_s = 1600 # material density [kg/m^3]
