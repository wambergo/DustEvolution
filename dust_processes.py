# load required packages, arguments and constants
import numpy as np
import time
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.integrate import ode
import scipy.integrate as spi

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

def relativehastigheder(a_i, a_j, r,z):
    """calculate relative velocities between particles

    :input a1: particle size 1
    :input a2: particle size 2
    :input r: the distance to central star from point in the center plane of the disc
    :input z: height above midplane 
    
    :return: H: The scale height in disc
    :return St_ij: Stokes number for particle 1 and 2
    :return all_vel: the total velocity
    """
    
    T = temp_funktion(r,False, False, False) # get temperature
   
    cs = np.sqrt(k*T/mu) # isothermal speed of sound
    kepler_frq = np.sqrt( np.abs((G*M_centralstar) / r**3 ) )
    kepler_vel =  kepler_frq * r
    H = cs / kepler_frq
    vn = (cs**2 / (2*kepler_vel)) * (delta+(7/4))
    
    sigma_d, rho_g, sigma_g, H_ud, kepler_frq_ud, cs_ud, T_ud = densitets_funktion(r) # load densities
    
    St_i = kepler_frq * ( (a_i * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) ) # Stokes-number for particle 1
    St_j = kepler_frq * ( (a_j * rho_s)/(cs * np.sqrt(8 / np.pi)*rho_g) ) # Stokes-number for particle 2
    
    vol_1 = (4/3) * np.pi * a_i**3 # volume of particle 1
    vol_2 = (4/3) * np.pi * a_j**3 # volume of particle 2

    m1 = vol_1*rho_s # [kg] mass of particle 1
    m2 = vol_2*rho_s # [kg] mass of particle 2

    delta_vb = np.sqrt((8*k*T*(m1+m2))/(np.pi*m1*m2))
    
    # Azimuthal motion
    delta_az = np.abs( vn * ( (1/(1+St_i**2)) - (1/(1+St_j**2)) ) )
    
    # Vertical settling    
    delta_vs = np.zeros(((len(a_i)), (len(a_j))))
    
    for i in range(0,len(St_i)):
        for j in range(0,len(St_j)):
            h_i = H * min(1, np.sqrt( alpha / ( min(St_i[i,j],0.5)*(1+St_i[i,j]**2) ) ) )
            h_j = H * min(1, np.sqrt( alpha / ( min(St_j[i,j],0.5)*(1+St_j[i,j]**2) ) ) )

            delta_vs[i,j] = np.abs(h_i * min(St_i[i,j], 0.5) - h_j * min(St_j[i,j], 0.5)) * kepler_frq
            
    # Turbulence
    vg = np.sqrt(alpha) * cs # turbulent gas velocity
    Re = alpha*sigma_g*sig_h2/(2*mu) # Reynolds-number approx 10^8
    
    St_eta = Re**(-1/2)
    ya = 1.6
    eps_turb = np.zeros((len(St_i),len(St_j)))
    
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

            # Regime I - closely coupled particles
            if St_i[i,j] < St_eta:
                term1_I = (St_i[i,j] - St_j[i,j]) / (St_i[i,j]  + St_j[i,j] )
                term2_I = (St_i[i,j]**2) / (St_i[i,j]  + Re**(-1/2))
                term3_I = (St_j[i,j]**2) / (St_j[i,j]  + Re**(-1/2))

                delta_vt[i,j]  = np.sqrt( vg**2 * term1_I * (term2_I - term3_I) )  
                            
            # Regime II - intermediate
            if St_eta <= St_i[i,j] and St_i[i,j] <= 1:
                term1_II = 2*ya
                term2_II = 1+eps_turb[i,j]
                term3_II = 2 / (1 + eps_turb[i,j])
                term4_II = 1 / (1 + ya)
                term5_II = eps_turb[i,j]**3 / (ya + eps_turb[i,j])

                delta_vt[i,j] = np.sqrt( vg**2 * (term1_II- term2_II + term3_II * (term4_II + term5_II)) * St_i[i,j] )
            
            # Regime III - heavy particles
            if St_i[i,j] > 1:
                term1_III = 1 / (1+St_i[i,j])
                term2_III = 1 / (1+St_j[i,j])
            
                delta_vt[i,j] = vg**2 * (term1_III + term2_III)
                delta_vt[i,j] = np.sqrt(delta_vt[i,j])
                
    
    # Radial drift
    vdust_i = (2*vn) / (St_i + (1/St_i) )
    vdust_j = (2*vn) / (St_j + (1/St_j) )

    v_gas = 3*alpha * ( (cs**2) / kepler_vel ) * ( (3/2) - delta)
    #Mdot = 2 * np.pi * r * Sigma * v_gas / Msun * year

    # Total
    v_totali = vdust_i + ( v_gas / (1+(St_i**2)) )
    v_totalj = vdust_j + ( v_gas / (1+(St_j**2)) )

    # Relative drift velocity
    delta_vd = np.abs(v_totali - v_totalj)
    return H, St_i, St_j, delta_vb, delta_az, delta_vs, delta_vt, delta_vd

def input_mass_distribution(a_min,a_max, a_maxp, n_a, r):
    """setup input mass distribution

    :input a_min: minimum particle size
    :input a_max: maximum particle size
    :input a_maxp: maximum particle size where there are particles to start with
    :input n_a: number of particle bins
    :input r: radial distance
    """
    
    # setup
    m_min = 4*np.pi / 3 * a_min**3 * rho_s
    m_max = 4*np.pi / 3 * a_max**3 * rho_s
    
    xi = 1.83
    
    mi = np.logspace(np.log10(m_min), np.log10(m_max), n_a+1)
    
    #mc = (mi[0:-2] + mi[1:-1]) / 2
    mc = np.zeros(n_a)
    for i in range(0,n_a):
        mc[i] = (mi[i]+mi[i+1])/2.
        
    ac = (3./(4.*np.pi) * mc / rho_s)**(1./3.)
    
    # determine normalizing constant alpha
    sigma_d, rho_g, sigma_g, H, kepler_frq, cs, T = densitets_funktion(r)
    
    mtilde = 0.
    
    for i in range(0,n_a):    
        if ac[i] < a_maxp:
            n_tilde = (mi[i+1]**(1.-xi) - mi[i]**(1.-xi))  / (1. - xi)
            mtilde += n_tilde*mc[i]
        
    alpha = sigma_d / mtilde
    
    nc = [ ]
    
    for i in range(0,n_a):
        if ac[i] < a_maxp:
            n_tilde = (mi[i+1]**(1.-xi) - mi[i]**(1.-xi))  / (1. - xi)
            nc.append(alpha*n_tilde)
        else:
            nc.append(0.)
    
    return mi, mc, nc, ac

def coagulation_kernel(a, m, r, motions_included='input'): 
    """setup coagulation kernel

    :input a: particle size
    :input m: particle mass
    :input r: radial distance
    
    :input motions included: 4 choices:
        * brownian alone 'brownian'
        * brownian with vertical settling 'bm_plus_sett'
        * brownian with vertical settling and azimuthal motion 'bm_pp_set_az'
        * all velocities included 'all'
    """
    
    # Get the relative velocities
    a_1 = a
    a_2 = a

    a_i,a_j = np.meshgrid(a_1,a_2)
    
    H, St_i, St_j, delta_vb, delta_vs, delta_az, delta_vt, delta_vd = relativehastigheder(a_i, a_j, r, 0.,) # z=0
    
    # fragmentation and coagulation probabilities
    psi = 1 # index
    
    vf = 10. # fragmentation velocity [m/s]
    
    # fragmentation probabilties
    if motions_included == 'brownian':
        pf = (delta_vb / vf)**psi * np.heaviside(vf-delta_vb,1) + np.heaviside(delta_vb - vf,1) 
        
    if motions_included == 'bm_plus_sett':
        delta_vbvs = np.sqrt(delta_vb**2 + delta_vs**2)
        pf = (delta_vbvs / vf)**psi * np.heaviside(vf-delta_vbvs,1) + np.heaviside(delta_vbvs - vf,1) 

    if motions_included == 'bm_pp_set_az':
        delta_vbsa = np.sqrt(delta_vb**2 + delta_vs**2 + delta_az**2)
        pf = (delta_vbsa / vf)**psi * np.heaviside(vf-delta_vbsa,1) + np.heaviside(delta_vbsa - vf,1)

    if motions_included == 'all':
        delta_all = np.sqrt(delta_vb**2 + delta_vs**2 + delta_az**2 + delta_vt**2 + delta_vd**2)
        pf = (delta_all / vf)**psi * np.heaviside(vf-delta_all,1) + np.heaviside(delta_all - vf,1)    
        
    # coagulation probability
    pc = 1 - pf
    
    # collision cross section
    sig_col = np.pi * (a_i + a_j)**2
    
    # for computing the verticale integration term
    hi = np.zeros((len(St_i),len(St_i))) # dust scale height
    hj = np.zeros((len(St_j),len(St_j))) # dust scale height
    
    # looping over stokes number
    for i in range(0,len(St_i)):
        for j in range(0,len(St_j)):
            hi[i,j] = H * min(1, np.sqrt( alpha / ( min(St_i[i,j],0.5)*(1+St_i[i,j]**2) ) ) )
            hj[i,j] = H * min(1, np.sqrt( alpha / ( min(St_j[i,j],0.5)*(1+St_j[i,j]**2) ) ) )
            
    vertical_int_term = 1 / np.sqrt(2 * np.pi*(hi**2 + hj**2))
    
    # koagulation kernels
    if motions_included == 'brownian':
        K_ij = delta_vb * sig_col * pc * vertical_int_term
        
    if motions_included == 'bm_plus_sett':
        K_ij = delta_vb_vs * sig_col * pc * vertical_int_term
        
    if motions_included == 'bm_pp_set_az':
        K_ij = delta_vbsa * sig_col * pc * vertical_int_term
        
    if motions_included == 'all':
        K_ij = delta_vtny * sig_col * pc * vertical_int_term
        
    # computing koagulation coefficient C
    def coag_koeff(m):
        C_ijk = np.zeros((n+1,n+1,n+1)) 

        for i in range(0,n):
            for j in range(0,n):
                mass_sum = m[i] + m[j] # letting masses coagulate
                for k in range(0,n):
                    if (mass_sum <= m[k+1]) and (mass_sum>=m[k]):
                        C_ijk[i,j,k] = (m[k+1]-mass_sum)/(m[k+1]-m[k])
                    if (mass_sum <= m[k]) and (mass_sum>=m[k-1]):
                        C_ijk[i,j,k] = (mass_sum-m[k-1])/(m[k]-m[k-1])
        
        return C_ijk
        
    C_ijk = coag_koeff(m)
    
    return K_ij, C_ijk

def ndot_coagulation(c,K,n):
    """running cogaulation eq
    """
    loss = -np.dot(K,n)*n
    
    gain=np.zeros(len(n))
    for i in range(len(n)):
        kcnn = (K*C[:,:,i])*n[:,np.newaxis]*n[np.newaxis,:]
        gain[i] = 0.5 * np.sum(kcnn)
        
    dndt = gain+loss

    return dndt

# Define Jacobian: jac[i,j] = d f[i] / d N[j], f[i] = dN[i]/dt
def jacobian(t,nc): 
    nn = len(nc)
    jac = np.zeros([nn,nn])
    for k in range(0,nn):
        for l in range(0,nn):
            jac[k,l] = np.sum(K[:,l] * C[:,l,k]*nc) - K[l,k]*nc[k]
    for k in range(0,nn):
        jac[k,k] -= np.sum(K[:,k]*nc)
    return jac

def ndot_coagulation2(t,nc):
    """puts equation system in proper form for scipy
    """
    ndot = ndot_coagulation(C,K,nc)
    tyr = t / year
    #print("t: " + str(tyr)) # printer tidsudvikling
    return ndot

n = 200 
z = 0

#r = 10 * AU
rr = np.array([1,10,100])*AU # distance array

a_min = 5*10**(-7) # 0.5 mu meter
a_maxp = 8*10**(-7) # 0.8 mu meter
a_max  = 5*10**(-2) # 5.0 cm

n_a = n + 1

tcurrent = 0.
t = tcurrent

tend = 1e6 * year
tend = 2e5 * year

tarr = (tcurrent, tend)

tstep = 1e5 * year

for r in rr:
    afstand = r/AU
    
    mi, mc, nc, ac = input_mass_distribution(a_min,a_max, a_maxp, n_a, r)
    nc = np.asarray(nc)
    sig_d0 = nc * mc

    K,C = coagulation_kernel(ac, mc, r,'bm_pp_set_az')

    ode = spi.ode(ndot_coagulation2,jacobian)

    # BDF method suitable to stiff systems of ODE's
    atol = 1e0 # absolute error
    rtol = 1e-6 # relative error
    with_jacobian = True
    ode.set_integrator('vode', method='bdf', with_jacobian=with_jacobian, nsteps=5000, atol=atol, rtol=rtol)
    ode.set_initial_value(nc, t)
    
    while ode.successful() and ode.t < tend:
        nf = ode.integrate(ode.t + tstep)

    plt.figure(1)
    plt.loglog()
    plt.plot(ac*1e2,nf*mc,label="Radial distance: %s AU" %afstand)
    plt.ylim(1e-19, 1e0)
    plt.title('Dust distribution, Brownian motion')
    plt.xlabel('Particle size [cm]')
    plt.ylabel('Surface density per size bin [$g \cdot cm^{-2}$]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
plt.show()
