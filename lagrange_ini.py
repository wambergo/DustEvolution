import numpy as np
from argumenter import r_in, r_yd, rho_s
from konstanter import year

n = 200
z = 0 # height above mid plane
a = 1e-4 # particle size

ri = np.logspace(np.log10(r_in), np.log10(r_yd), n+1)
rl = ri[0:-2] # radius-bin upper bound
ru = ri[1:-1] # radius-bin lower bound
rc = 0.5 * (rl+ru) # radius bin central
drc = ru - rl

r = rc
dr = drc

tcurrent = 0.
t = [tcurrent]
tend = 1e6 * year
tdelta = 2e5 # tdelta measured in yr

mass_s = 4 * np.pi / 3 * a**3 * rho_s # dust particle mass
