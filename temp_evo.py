# load required packages, arguments and constants
import numpy as np
import time 

from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from argumenter import phi, Tstar, M_centralstar, R, nr, r_in, r_yd, dlnHpdlr, alpha, delta, rho_s
from konstanter import AU, year, k, mu, sig_h2, sigma_sb, G, Msun
from scipy.integrate import quad

def densitets_funktion(r):
    """setup densities

    :input r the distance to central star from point in the center plane of the disc
    :return values for gas and dusts surface and mass densities
    """
    T = 0.05**(1/4)*np.abs(r/R)**(-1/2)*Tstar
    cs = np.sqrt(k*T/mu) # isotermisk lydhastighed
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ))
    H = cs / kepler_frq
    
    def integrand(r):
        return r*(r/AU)**-delta
    
    cond_array = quad(integrand, r_in, r_yd)
    cond = 2*np.pi*cond_array[0] # lhs af betingelse for sigma0
    M_disk = 0.01 * Msun # disk massen [kg]
    Sigma0 = M_disk / cond

    sigma_g = Sigma0 * np.abs((r/AU))**-delta # overflade densitet
    rho_g = (sigma_g / (np.sqrt(2*np.pi)*H)) * np.exp((-z**2)/(2*H**2))
    epsilon=0.01 # dust2gas
    sigma_d = sigma_g * epsilon
    
    return sigma_d, rho_g, sigma_g, H, kepler_frq, cs, T

def temp_funktion(r,int_plot=True, tmid_plot=True, one_particle_size=False):  
    """calculate temperature in midplane

    :input r the distance to central star from point in the center plane of the disc
    :return temperature in midplane
    """
    
    # load Rosseland and Planck average opacities
    kR_data = np.loadtxt('kR.csv') # Rosseland avr opacities
    kP_data = np.loadtxt('kP.csv') # Planck avr opacities

    T_R = kR_data[:,0]
    rho_R = kR_data[:,1]
    kappa_R = kR_data[:,2]

    T_P = kP_data[:,0]
    rho_P = kP_data[:,1]
    kappa_P = kP_data[:,2]
    
    # interpolation
    interpolering_funktionR = interpolate.bisplrep(T_R,rho_R,kappa_R,w=None, task=0, s=27907)
    interpolering_funktionP = interpolate.bisplrep(T_P,rho_P,kappa_P,w=None, task=0, s=48875)

    T_gitterR = np.unique(T_R) # temperature grid values
    rho_gitterR = np.unique(rho_R) # density grid values

    T_gitterP = np.unique(T_P)
    rho_gitterP = np.unique(rho_P)

    kappa_approxR = interpolate.bisplev(T_gitterR, rho_gitterR, interpolering_funktionR)
    kappa_approxP = interpolate.bisplev(T_gitterP, rho_gitterP, interpolering_funktionP)
    
    # plotting
    if int_plot:
        plt.figure()
        plt.title("Rosseland avr dust opacities")
        plt.xlabel("Temperature [K]")
        plt.ylabel(r'$\rho_g \quad [g/cm^3] $')
        plt.pcolor(T_gitterR, rho_gitterR, kappa_approxR,cmap='RdGy')
        plt.colorbar(label='$\kappa_R \quad [cm^2/g]$')
        plt.show()

        plt.figure()
        plt.title("Planck avr dust opacities")
        plt.xlabel("Temperature [K]")
        plt.ylabel(r'$\rho_g \quad [g/cm^3] $')
        plt.pcolor(T_gitterP, rho_gitterP, kappa_approxP,cmap='RdGy')
        plt.colorbar(label='$\kappa_R \quad [cm^2/g]$')
        plt.show()
        
    sigma_d, rho_g, sigma_g, H, kepler_frq, cs, T = densitets_funktion(r)
    
    T_unik = np.unique(T)
    rho_unik = np.unique(rho_g)
    
    newkappa_R = interpolate.bisplev(T_unik, rho_unik, interpolering_funktionR)
    newkappa_P = interpolate.bisplev(T_unik, rho_unik, interpolering_funktionP)
    
    tau_R = 0.5 * sigma_d * newkappa_R * 0.1
    tau_P = 0.5 * sigma_d * newkappa_P * 0.1
    
    print(tau_R)
    if one_particle_size:
        tau_R = tau_R[0]
        tau_P = tau_P[0]
    else:
        tau_R = tau_R
        tau_P = tau_P
    
    # Calculation of the temperature contribution from star and external radiation
    T_irr = Tstar * ( (2/(3*np.pi)) * np.abs(R/r)**3 + (1/2) * (R/r)**2 * (H/r) * (dlnHpdlr-1) )**(1/4)
    
    # Calculation of the temperature in the midplane
    first_term = (3/8)*tau_R
    second_term = 1/(2*tau_P)   
    
    T_mid = ( (9/(8*sigma_sb)) * (first_term+second_term)*sigma_g*alpha*(cs**2)*kepler_frq+ T_irr**4)**(1/4)
    
    if tmid_plot:
        plt.figure()
        plt.semilogx(r/AU, T_mid[0], 'k')
        plt.title("Temperatur in midplane")
        plt.xlabel('r [AU]')
        plt.ylabel('T [k]')
        plt.grid(True)
        
    return T_mid
