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

a_1 = np.logspace(-6, 2, 100) # partikel 1 radius
a_2 = np.logspace(-6, 2, 100) # partikel 2 radius
r = 1*AU # afstand til central stjerne fra punkt i midtplanen
z = 0

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
        plt.title("Rosseland avr dust opacities")
        plt.xlabel("Temperatur [K]")
        plt.ylabel(r'$\rho_g \quad [g/cm^3] $')
        plt.pcolor(T_gitterR, rho_gitterR, kappa_approxR,cmap='RdGy')
        plt.colorbar(label='$\kappa_R \quad [cm^2/g]$')
        plt.show()
        plt.figure()
        plt.title("Planck avr dust opacities")
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
        plt.semilogx(r/AU, T_mid, 'k')
        plt.title("Temperature in midplane")
        plt.xlabel('r [AU]')
        plt.ylabel('T [k]')
        plt.grid(True)
    return T_mid


def brownian_vel(a_1, a_2, r,z):
    """
    Brownian velocites
    """
    T = temp_funktion(r,False, False, False)
    a_i,a_j = np.meshgrid(a_1,a_2)
    ai_cm = np.logspace(-4, 4, 100)
    aj_cm = np.logspace(-4, 4, 100)
    x, y = np.meshgrid(ai_cm,aj_cm)
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) # vol_1 = (4/3) * np.pi * a_i**3 # 
    vol_1 = (4/3) * np.pi * a_i**3 
    vol_2 = (4/3) * np.pi * a_j**3 
    m1 = vol_1*rho_s 
    m2 = vol_2*rho_s 
    delta_vb = np.sqrt((8*k*T*(m1+m2))/(np.pi*m1*m2))
    
    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
    
    plt.loglog()
    plt.title("Brownian motion")
    plt.xlabel("Particle $a_i$ radius [cm]")
    plt.ylabel("Particle $a_j$ radius [cm]")
    plt.contourf(x,y, delta_vb, 50, cmap='RdGy')
    plt.colorbar(label="Relative velocities [m/s]",format=ticker.FuncFormatter(fmt))
    plt.show()
    return delta_vb

def rad_vel(a_1, a_2, r,z):
    """
    Radial velocites
    """
    T = temp_funktion(r,False, False, False)

    cs = np.sqrt(k*T/mu) 
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel = kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))

    a_i,a_j = np.meshgrid(a_1,a_2)
    ai_cm = np.logspace(-4, 4, 100)
    aj_cm = np.logspace(-4, 4, 100)
    x, y = np.meshgrid(ai_cm,aj_cm)
    
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r)
    
    St_i = kepler_frq * ( (a_i * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    St_j = kepler_frq * ( (a_j * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    
    vdust_i = (2*vn) / (St_i + (1/St_i) )
    vdust_j = (2*vn) / (St_j + (1/St_j) )

    v_gas = 3*alpha * ( (cs**2) / kepler_vel ) * ( (3/2) - delta)
    
    v_totali = vdust_i + ( v_gas / (1+(St_i**2)) )
    v_totalj = vdust_j + ( v_gas / (1+(St_j**2)) )
    
    delta_vd = np.abs(v_totali - v_totalj)
    
    plt.loglog()
    plt.title("Radial inflow")
    plt.xlabel("Particle $a_i$ radius [cm]")
    plt.ylabel("Particle $a_j$ radius [cm]")
    plt.contourf(x,y, delta_vd, 50, cmap='RdGy')
    plt.colorbar(label="Relative velocties [m/s]")
    plt.show()
    return delta_vd

def azi_vel(a_1, a_2, r,z):
    """
    Azimuthal inflow
    """
    T = temp_funktion(r,False, False, False)
    cs = np.sqrt(k*T/mu)
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel = kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    
    a_i,a_j = np.meshgrid(a_1,a_2)
    ai_cm = np.logspace(-4, 4, 100)
    aj_cm = np.logspace(-4, 4, 100)
    x, y = np.meshgrid(ai_cm,aj_cm)
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r)
    
    St_i = kepler_frq * ( (a_i * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    St_j = kepler_frq * ( (a_j * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    
    delta_az = np.abs( vn * ( (1/(1+St_i**2)) - (1/(1+St_j**2)) ) )
    
    plt.loglog()
    plt.title("Azimuthal inflow")
    plt.xlabel("Particle $a_i$ radius [cm]")
    plt.ylabel("Particle $a_j$ radius [cm]")
    plt.contourf(x, y, delta_az, 50, cmap='RdGy')
    plt.colorbar(label="[m/s]")
    plt.show()
    return delta_az

def turb_vel(a_1, a_2, r,z):
    """
    Turbulence
    """
    T = temp_funktion(r,False, False, False)
        
    cs = np.sqrt(k*T/mu) 
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel =  kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    
    a_i,a_j = np.meshgrid(a_1,a_2)
    
    ai_cm = np.logspace(-4, 4, 100)
    aj_cm = np.logspace(-4, 4, 100)
    x, y = np.meshgrid(ai_cm,aj_cm)
    
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) 

    St_i = kepler_frq * ( (a_i * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    St_j = kepler_frq * ( (a_j * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) )
    
    vg = np.sqrt(alpha) * cs 
    beta = (H/r)**2
    Re = alpha*sigma_g*sig_h2/(2*mu) 
    
    St_eta = Re**(-1/2)
    ya = 1.6
    eps_turb = np.zeros((len(St_i),len(St_j)))

    #St_ij = (1+np.sqrt((3*beta)/(2*alpha)))**(-1/2)

    delta_vt = np.zeros((len(St_i),len(St_j)))
    
    for i in range(len(St_i)):
        for j in range(len(St_j)):
        
            if i >= j:
                St_i[i,j] = St_i[i,i]
                St_j[i,j] = St_j[j,j]

            else:
                St_i[i,j] = St_i[j,j]
                St_j[i,j] = St_j[i,i]

            eps_turb = St_j / St_i # stopping time ratio

            # Regime I 
            if St_i[i,j] < St_eta:
                term1_I = (St_i[i,j] - St_j[i,j]) / (St_i[i,j]  + St_j[i,j] )
                term2_I = St_i[i,j] **2 / (St_i[i,j]  + Re**(-1/2))
                term3_I = St_j[i,j] **2 / (St_j[i,j]  + Re**(-1/2))

                delta_vt[i,j]  = np.sqrt( vg**2 * term1_I * (term2_I - term3_I) )  
                            
            # Regime II 
            if St_eta <= St_i[i,j] and St_i[i,j] <= 1:
                term1_II = 2*ya
                term2_II = 1+eps_turb[i,j]
                term3_II = 2 / (1 + eps_turb[i,j])
                term4_II = 1 / (1 + ya)
                term5_II = eps_turb[i,j]**3 / (ya + eps_turb[i,j])

                delta_vt[i,j] = np.sqrt( vg**2 * (term1_II- term2_II + term3_II * (term4_II + term5_II)) * St_i[i,j] )
            
            # Regime III 
            if St_i[i,j] > 1:
                term1_III = 1 / (1+St_i[i,j])
                term2_III = 1 / (1+St_j[i,j])
            
                delta_vt[i,j] = vg**2 * (term1_III + term2_III)
                delta_vt[i,j] = np.sqrt(delta_vt[i,j])
    
    plt.loglog()
    plt.title("Turbulence")
    plt.xlabel("Particle $a_i$ radius [cm]")
    plt.ylabel("Particle $a_j$ radius [cm]")
    plt.contourf(x, y, delta_vt, 50, cmap='RdGy')
    plt.colorbar(label="[m/s]")
    plt.show()
    
    return delta_vt

def udf_vel(a_1, a_2, r, z):
    """
    Vertical settling
    """
    T = temp_funktion(r,False, False, False)

    cs = np.sqrt(k*T/mu) 
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel =  kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    
    a_i,a_j = np.meshgrid(a_1,a_2)
    
    ai_cm = np.logspace(-4, 4, 100)
    aj_cm = np.logspace(-4, 4, 100)
    x, y = np.meshgrid(ai_cm,aj_cm)
    
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) 

    St_i = kepler_frq * ( (a_i * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) ) 
    St_j = kepler_frq * ( (a_j * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) ) 
    
    # initialiserer
    delta_vs = np.zeros(((len(a_1)), (len(a_2) ) ) )

    for i in range(0,len(St_i)):
        for j in range(0,len(St_j)):
            h_i = H * min(1, np.sqrt( alpha / ( min(St_i[i,j],0.5)*(1+St_i[i,j]**2) ) ) )
            h_j = H * min(1, np.sqrt( alpha / ( min(St_j[i,j],0.5)*(1+St_j[i,j]**2) ) ) )

            delta_vs[i,j] = np.abs(h_i * min(St_i[i,j], 0.5) - h_j * min(St_j[i,j], 0.5)) * kepler_frq
               
    plt.loglog()
    plt.title("Vertical settling")
    plt.xlabel("Particle $a_i$ radius [cm]")
    plt.ylabel("Particle $a_j$ radius [cm]")
    plt.contourf(x, y, delta_vs, 50, cmap='RdGy')
    plt.colorbar(label="[m/s]")
    plt.show()
    
    return delta_vs

# Total relative velocities
vb = brownian_vel(a_1, a_2, r,z)
vd = rad_vel(a_1, a_2, r,z)
az = azi_vel(a_1, a_2, r,z)
vt = turb_vel(a_1, a_2, r,z)
vs = udf_vel(a_1, a_2, r, z)

all_vel = (vd**2 + vt**2 + vb**2 + az**2 + vs**2)**0.5

a_i,a_j = np.meshgrid(a_1,a_2)

ai_cm = np.logspace(-4, 4, 100)
aj_cm = np.logspace(-4, 4, 100)
x, y = np.meshgrid(ai_cm,aj_cm)

plt.loglog()
plt.title("Relative particle velocities")
plt.xlabel("Particle $a_i$ radius [cm]")
plt.ylabel("Particle $a_j$ radius [cm]")
plt.contourf(x, y, all_vel, 50, cmap='RdGy')
plt.colorbar(label="[m/s]")
plt.show()

plt.loglog()
plt.title("Relative particle velocities")
plt.xlabel("Particle $a_i$ radius [cm]")
plt.ylabel("Particle $a_j$ radius [cm]")
contours = plt.contour(x, y, all_vel, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.show()
