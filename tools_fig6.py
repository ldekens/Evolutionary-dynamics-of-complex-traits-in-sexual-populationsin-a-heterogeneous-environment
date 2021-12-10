#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:49:42 2021

@author: dekens
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:19:54 2021

@author: dekens
"""
import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin
import scipy.signal as scsign
import os
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Pool
from itertools import repeat

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "text.usetex": True})
viridis = cm.get_cmap('viridis', 300)


####### Creates grids used in the reproduction operator (double convolution - see Appendix G)
def grid_double_conv(zmax, Nz):
    z, dz = np.linspace(-zmax, zmax, Nz, retstep=True)
    zz, dzz = np.linspace(-zmax*2, zmax*2, 2*Nz-1, retstep=True)
    z2, dz2 = np.linspace(-zmax*2, zmax*2, 4*Nz-3, retstep=True)
    
    return(Nz, z, dz, zz, dzz, z2, dz2)

####### Creates a discritized Gaussian distribution with mean m and variance s**2 on the grid z
def Gauss(m, s, z):
    Nz = np.size(z)
    G = np.zeros(Nz)
    for k in range(Nz):
        G[k] = 1/( np.sqrt(2*np.pi)*s )* np.exp( - (z[k]-m)**2 / (2*s**2) )
    return(G)
    
####### Encodes the reproduction operator - double convolution
def reproduction_conv(n1, n2, N, epsilon, zmax, Nz):
    Nz, z, dz, zGauss, dzGauss, zaux, dzaux = grid_double_conv(zmax, Nz)

    Gsex = Gauss(0, epsilon/np.sqrt(2), 0.5*zGauss)
    Bconv_aux = scsign.convolve(scsign.convolve(n1, Gsex*dz)*dz, n2)
    if (N>0):
        Bconv = np.interp(z, zaux, Bconv_aux)/N
    else:
        Bconv = np.zeros(np.size(z))
    return(Bconv)

####### Create a directory of path workdir
def create_directory(workdir):
    try:
        os.mkdir(workdir)
        print("Directory " , workdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , workdir ,  " already exists")
    return
    

####### Function that implements the scheme iterations (see Appendix G for details)
def update(n1, n2, parameters, M_semi_implicit_scheme, dt, dz, Nz, zmax, err):
    epsilon, r, kappa, theta = parameters
    
    #### Scheme of our model
    N1, N2 = sum(n1*dz), sum(n2*dz)
    # Reproduction terms
    B1, B2 = reproduction_conv(n1, n1, N1, epsilon, zmax, Nz), reproduction_conv(n2, n2, N2, epsilon, zmax, Nz)
    B12 = np.concatenate((B1, B2))
    n12aux = np.concatenate((n1, n2))
    N12 = np.concatenate((np.ones(Nz)*N1, np.ones(Nz)*N2))
    n12 = scisplin.spsolve((M_semi_implicit_scheme + dt/(epsilon**2)*sp.spdiags(kappa*N12, 0, 2*Nz, 2*Nz)),(n12aux + dt/(epsilon**2)*r*B12))
    n1new = n12[:Nz]
    n2new = n12[Nz:]
    
    #### calculating the discrepancy term between past and present state
    err=np.maximum(max(abs((n1new - n1)/dt)), max(abs((n2new - n2)/dt)))
    
    return(n1new, n2new, err)

####### Compute the score attributed to the final state of a simulation according to the formula in Appendix I
def scoring(n1f, n2f, z, dz, epsilon):
    N1f, N2f = sum(n1f*dz), sum(n2f*dz)
    nf = n1f + n2f
    z_mean = sum(z*nf)*dz/(N1f+N2f)
    var = sum((z-z_mean)**2*nf)*dz/(N1f+N2f)
    outcome = 0
    if (max(N1f, N2f) > 0.01):
        if (var < 2*epsilon**2):
            outcome = 5/6-abs(N1f-N2f)/(N1f+N2f)*1/3
        else:
            outcome = 1
    return(outcome)

def run_simulation(n1initial, n2initial, m, g, parameters, T, Nt, dt, zmax, z, Nz, dz, errmax):
    epsilon, r, kappa, theta = parameters
    
    #### Auxiliary matrices
    #Migration
    Mmigration = sp.coo_matrix(np.block([[-np.eye(Nz)*m, np.eye(Nz)*m],[np.eye(Nz)*m,-np.eye(Nz)*m]]))
    
    #selection
    Vselection1 = -g*(z+theta)*(z+theta)
    Vselection2 = -g*(z-theta)*(z-theta)
    Vselection = np.concatenate((Vselection1, Vselection2))
    Mselection = sp.spdiags(Vselection, 0, 2*Nz, 2*Nz)
    
    Id = sp.spdiags(np.ones(2*Nz), 0, 2*Nz, 2*Nz)
    M_semi_implicit_scheme = Id - dt/(epsilon**2)*(Mmigration + Mselection)
    
    n1, n2, err = n1initial, n2initial, 10
    
    for t in T:
        n1, n2, err = update(n1, n2, parameters, M_semi_implicit_scheme, dt, dz, Nz, zmax, err)
    if (err > errmax):
        while (t < T[-1])&(err > errmax):
           n1, n2, err = update(n1, n2, parameters, M_semi_implicit_scheme, dt, dz, Nz, zmax, err)
      
    outcome = scoring(n1, n2, z, dz, epsilon)
    print(m)
    print(g)
    print(outcome)
    return(outcome)

####### Auxiliary function that runs simulations for a fixed selection parameter g (for parallel processing)
def run_simulations_g(g, n1initial, n2initial, M, parameters, T, Nt, dt, zmax, z, Nz, dz, errmax):
    output_g = np.zeros(np.size(M))
    for j in range(np.size(M)):
        m = M[j]
        output_g[j] = run_simulation(n1initial, n2initial, m, g, parameters, T, Nt, dt, zmax, z, Nz, dz, errmax)
    print(g)
    return(output_g)

####### Displays the comparison between a moment time series deduced from our scheme and RK 2001's one (size and mean)
def plot_summary_outcomes(outcomes, title):
    ml.rcParams['mathtext.fontset'] = 'stix'
    ml.rcParams['font.family'] = 'STIXGeneral'
    plt.figure(figsize=(10, 10))
    plt.imshow(outcomes, cmap='viridis', vmin = 0, vmax = 1, origin='lower', extent=(0, 3, 0, 3))
    plt.axis('equal')
    plt.xlabel('Intensity of selection $(g)$', fontsize=40)
    plt.ylabel('Migration rate $(m)$', fontsize=40)
    plt.yticks(fontsize=20)
    plt.xticks(ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=20)
    plt.savefig(title + '.png')
    plt.show()


######## Function that runs the whole scheme, given the initial distributions n1 and n2 and the initial moments (RK)
######## Produces final time series comparison between the moments generated by the two models and save them 
def run_outcomes(n1initial, n2initial, parameters, T, Nt, dt, zmax, z, Nz, dz, title, workdir):
    ## Migration rates and selection strengths
    Nm, Ng = 60, 60
    M = np.linspace(0.01, 3, Nm)
    G = np.linspace(0.01, 3, Ng)
    
    outcomes = np.zeros((Nm, Ng))
    errmax = 1e-5
    
    p  = Pool(6)
    inputs = [*zip(G, repeat(n1initial), repeat(n2initial), repeat(M), repeat(parameters), repeat(T), repeat(Nt), repeat(dt), repeat(zmax), repeat(z), repeat(Nz), repeat(dz), repeat(errmax))]
    outcomes = np.transpose(p.starmap(run_simulations_g, inputs))
    
    
    np.save(title, outcomes)
    plot_summary_outcomes(outcomes, title)
        