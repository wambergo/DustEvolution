# load required packages, arguments and constants
import numpy as np
import time
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pylab as pl

# some setup for visualizing
from brewer2mpl import get_map
cols = get_map('RdGy','Diverging',9).mpl_colors
col1 = cols[0]
col2 = cols[1]
col3 = cols[2]
col4 = cols[3]
col5 = cols[4]
col6 = cols[5]
col7 = cols[6]
col8 = cols[7]
col9 = cols[8]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[col1, col6 , col3, col4, col5, col7, col8, col9])

from argumenter import phi, Tstar, M_centralstar, R, nr, r_in, r_yd, dlnHpdlr, alpha,delta, rho_s
from konstanter import AU, year, k, mu, sig_h2, sigma_sb, G, Msun
from lagrange_ini import n, z, a, ri, rl, ru, rc, drc, r, dr, tcurrent, t, tend, tdelta, mass_s

def densitets_funktion(r):
    """setup densities

    :input: r the distance to central star from point in the center plane of the disc
    :return: values for gas and dusts surface and mass densities
    """
    T = 0.05**(1/4)*np.abs(r/R)**(-1/2)*Tstar
    cs = np.sqrt(k*T/mu) # isothermal speed of sound
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ))
    H = cs / kepler_frq
    
    def integrand(r):
        return r*(r/AU)**-delta

    cond_array = quad(integrand, r_in, r_yd)
    cond = 2*np.pi*cond_array[0] # lhs of condition for sigma0
    M_disk = 0.02 * Msun # disc mass [kg]
    Sigma0 = M_disk / cond
    sigma_g = Sigma0 * np.abs((r/AU))**-delta # surface density
    rho_g = (sigma_g / (np.sqrt(2*np.pi)*H)) * np.exp((-z**2)/(2*H**2))
    epsilon=0.01 # dust2gas
    sigma_d = sigma_g * epsilon
    return sigma_d, rho_g, sigma_g, H, kepler_frq, cs, T

def temp_funktion(r,int_plot=True, tmid_plot=True, one_particle_size=True):
    """calculate temperature in midplane

    :input: r the distance to central star from point in the center plane of the disc
    :return: temperature in midplane
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
    interpolering_funktionR = interpolate.bisplrep(T_R,rho_R,kappa_R,w=None, task=0,s=27907)
    interpolering_funktionP = interpolate.bisplrep(T_P,rho_P,kappa_P,w=None, task=0,s=48875)
    T_gitterR = np.unique(T_R) # temperature grid values
    rho_gitterR = np.unique(rho_R) # density grid values
    T_gitterP = np.unique(T_P)
    rho_gitterP = np.unique(rho_P)
    
    kappa_approxR = interpolate.bisplev(T_gitterR, rho_gitterR, interpolering_funktionR)
    kappa_approxP = interpolate.bisplev(T_gitterP, rho_gitterP, interpolering_funktionP)
                                                                                                    
    if int_plot:
        plt.figure()
        plt.title("Rosseland gns. dust opacities")
        plt.xlabel("Temperatur [K]")
        plt.ylabel(r'$\rho_g \quad [g/cm^3] $')
        plt.pcolor(T_gitterR, rho_gitterR, kappa_approxR,cmap='RdGy')
        plt.colorbar(label='$\kappa_R \quad [cm^2/g]$')
        plt.show()
        plt.figure()
        plt.title("Planck gns. dust opacities")
        plt.xlabel("Temperatur [K]")
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
        plt.title("Temperature in midplane")
        plt.xlabel('r [AU]')
        plt.ylabel('T [k]')
        plt.grid(True)
    return T_mid

def hastighed_funktion(r,z,a):
    """calculate radial velocity

    :input r: the distance to central star from point in the center plane of the disc
    :input z: height above the center plane of the disc
    :input a: radius of particle
    :return: radial velocity
    """
    T = temp_funktion(r,False, False)
    
    # calculate max drift velocity
    cs = np.sqrt(k*T/mu) # isothermal speed of sound
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel = kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) # get densities
    
    # calculate stokes number
    St = kepler_frq * ( (a * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )

    # calculate the two sources to radial drift
    vdust = -(2*vn) / (St + (1/St) )
    v_gas = -3*alpha * ( (cs**2) / kepler_vel ) * ( (3/2) - delta)
    Mdot = 2 * np.pi * r * sigma_g * v_gas / Msun * year
    # total contribution
    vr = vdust + ( v_gas / (1+(St**2)) )
    return vr

def hastighed_funktion(r,z,a):
    """calculate radial velocity

    :input r: the distance to central star from point in the center plane of the disc
    :input z: height above the center plane of the disc
    :input a: radius of particle
    :return: radial velocity
    """
    T = temp_funktion(r,False, False)
    
    # calculate max drift velocity
    cs = np.sqrt(k*T/mu) # isothermal speed of sound
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel = kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) # get densities
    
    # calculate stokes number
    St = kepler_frq * ( (a * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )

    # calculate the two sources to radial drift
    vdust = -(2*vn) / (St + (1/St) )
    v_gas = -3*alpha * ( (cs**2) / kepler_vel ) * ( (3/2) - delta)
    Mdot = 2 * np.pi * r * sigma_g * v_gas / Msun * year
    # total contribution
    vr = vdust + ( v_gas / (1+(St**2)) )
    return vr

def evolve(ri, drc, N_bin, t, tcurrent, tend, z, a, metode, tdelta, plot_evolve=True):
    """Moving some dust

    :input ri: initial radius bin
    :input drc: central step initial
    :input N_bin: initial # bin
    :input t: inital time for evolution
    :input tcurrent: current time
    :input tend: time for whole simulation
    :input z: height in disc
    :input a: particle radius
    :input metode: method 1) Euler 2) Lagrange
    """
    
    tid = time.time() # tracking time for simulation
    
    tnext = 0.
    
    if metode == 2:
        rl = ri[0:-2] # radius-bin upper bound
        ru = ri[1:-1] # radius-bin lower bound
        rc = 0.5 * (rl+ru) # radius bin central
        drc = ru - rl
        n = np.size(rc)
        vr = hastighed_funktion(ri,z,a)
        vrc = 0.5 * (vr[0:-2]+vr[1:-1])
        dt = min(0.2 * np.abs(drc / vrc))
        #print(dt)
        ri_new = ri + dt*vr
        
    # Upper boundary condition at interface item. n with constant influx of new material
    sigma_d_ub, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(ri[n]+0.5*drc[n-1])
    drub = -vr[n] * dt
    
    ##############################################
    # Calculate fractional transport for all cells
    ##############################################
    # Lower boundary condition. Outflow of material only
    frac = []
    frac.append((ri[0]**2 - ri_new[0]**2) / (ri_new[1]**2 - ri_new[0]**2))
    
    for i in range(1,n):
        if vr[i] < 0: # dust particles moving in
            frac.append((ri[i]**2 - ri_new[i]**2) / (ri_new[i+1]**2 - ri_new[i]**2))
        if vr[i] > 0: #  dust particles moving out
            frac.append((ri[i]**2 - ri_new[i]**2) / (ri_new[i]**2 - ri_new[i-1]**2))

    while (tcurrent < tend):
        if metode == 1: # Lagrange method
            vr = hastighed_funktion(ri,z,a)
            vrc = 0.5 * (vr[0:-2]+vr[1:-1])
            n = np.size(drc)
            
            for i in range(0,n-1):
                # makes sure the ring does not fall into the central star
                if ri[i] < r_in:
                    break
                
            dt = min(0.2 * np.abs(drc[0:i] / vrc[0:i]))
            
            vr[i+1:-1] = 0.
            ri = ri + dt * vr
            # update derived radius variable
            rl = ri[1:-1]
            ru = ri[0:-2]
            rc = 0.5 * (rl+ru)
            drc = ru - rl
            tcurrent = tcurrent + dt
            t.append(tcurrent)
            
            for i in range(0,n-1):
                # makes sure the ring does not fall into the central star
                if rc[i] < r_in:
                    break
                    
            i = min(i,n-1)
            #print(i,n,np.size(ri))
            r_final = rc[0:i]
            dr_final = drc[0:i]
            # Calculate number density and surface density
            n_d = N_bin[0:i] / (2 * np.pi * r_final * dr_final)
            sigma_d = n_d * mass_s
            
            # Plotting
            t_f = t[-1] / year # convert to years
            
            if (t_f > tnext):
                tnext += tdelta
                print("Plotting at t=",t_f)
                plt.figure(1)
                plt.subplot(211)
                plt.loglog()
                plt.title("Lagrange method")
                plt.xlabel("r [AU]")
                plt.ylabel("$n_{dust}$")
                plt.plot(r_final / AU, n_d*0.1, label="time: %.1E yr" %t_f)
                plt.legend()
                plt.subplot(212)
                plt.loglog()
                plt.xlabel("r [AU]")
                plt.ylabel("$ \Sigma_d \quad [g/cm^2]$")
                plt.plot(r_final / AU, sigma_d*0.1, label="time: %.1E yr" %t_f)
                plt.legend()
                
        if metode == 2:
            N_new = N_bin # initializing

            tcurrent = tcurrent + dt
            t.append(tcurrent)
                        
            # Lower bound. Only allows outflows no matter what
            if (vr[0] < 0):
                delta_N = N_bin[0] * frac[0]
                N_new[0] = N_new[0] - delta_N

            for i in range(1,n):
                if vr[i] < 0: # dust particles moving in
                    delta_N = N_bin[i] * frac[i]
                        
                if vr[i] > 0: # dust particles moving in
                    delta_N = N_bin[i-1] * frac[i]

                N_new[i-1] += delta_N
                N_new[i]   -= delta_N
            
            # Upper boundary condition at interface item. n with constant influx of new material
            N_new[n-1] += sigma_d_ub / mass_s * np.pi * ((ri[n] + drub)**2 - ri[n]**2)

            N_bin = N_new
            tau_drift = rc / np.abs(vrc)
            tau_drift_yr = tau_drift / year
            #print("tau: " + str(tau_drift_yr))
            
            # Plotting
            t_f = t[-1] / year # converting to years
            if (t_f > tnext):
                tnext += tdelta
                print("Plotting at t=",t_f)
                n_d = N_bin / (2 * np.pi * rc * drc)
                sigma_d = n_d * mass_s        

                plt.figure(1)
                plt.subplot(211)
                plt.loglog()
                plt.title("Evolution of the dust surface density")
                plt.xlabel("r [AU]")
                plt.ylabel("$n_{dust}$")
                plt.plot(rc / AU, n_d*0.1, label="time: %.1E yr" %t_f)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                plt.subplot(212)
                plt.loglog()
                plt.xlabel("r [AU]")
                plt.ylabel("$ \Sigma_d \quad [g/cm^2]$")
                plt.plot(rc / AU, sigma_d*0.1, label="time: %.1E yr" %t_f)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

          
    rc_test = rc / AU
    return vrc, rc_test, tau_drift_yr

def main():
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(rc) # load densities
    N_bin = sigma_d / mass_s * 2 * np.pi * rc * dr # number density @ t=0

    # LETS MOVE SOME DUST
    evolve(ri, drc, N_bin, t, tcurrent, tend, z, a, 2, tdelta)
    
if __name__ == "__main__":
    main()
