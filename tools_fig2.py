#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:19:54 2021

@author: dekens
"""
import numpy as np
import scipy.linalg as scilin
from scipy import sparse as sp
import scipy.sparse.linalg as scisplin
import scipy.signal as scsign
import os
import matplotlib as ml
import matplotlib.pyplot as plt
ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "text.usetex": True})
    
####### Function that translates the parameters from RK(2001) to those used in our analysis
def equivalence_RK2001_parameters(parameters_RK):
    sigmag, rstar, Kstar, gamma, m, theta1, theta2 = parameters_RK
    r0, K = rstar + gamma/2*sigmag**2, Kstar*(rstar + gamma/2*sigmag**2)/rstar
    return(sigmag, r0, r0/K, gamma/2, m, (theta2-theta1)/2)

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

####### Compute the first four moments of a given a discrete distribution n on grid z and step dz
def moments(n, z, dz):
    N = sum(n)*dz
    m = sum(z*n)*dz/N
    v = sum((z - m)**2*n)*dz/N
    s = sum((z - m)**3/np.sqrt(v)**3*n)*dz/N
    return(N, m , v, s)
    

####### Function that implements the scheme iterations (see Appendix G for details)
def update(n1, n2, RK, parameters, parameters_RK, M_semi_implicit_scheme, V_RK, dt, dz, Nz, zmax):
    epsilon, r, kappa, g, m, theta = parameters
    sigmag, rstar, Kstar, gamma, m, theta1, theta2 = parameters_RK
    r0, K = rstar + gamma/2*sigmag**2, Kstar*(rstar + gamma/2*sigmag**2)/rstar
    
    #### Scheme of our model
    N1, N2 = sum(n1*dz), sum(n2*dz)
    # Reproduction terms
    B1, B2 = reproduction_conv(n1, n1, N1, epsilon, zmax, Nz), reproduction_conv(n2, n2, N2, epsilon, zmax, Nz)
    B12 = np.concatenate((B1, B2))
    n12aux = np.concatenate((n1, n2))
    N12 = np.concatenate((np.ones(Nz)*N1, np.ones(Nz)*N2))
    n12 = scisplin.spsolve((M_semi_implicit_scheme + dt/(epsilon**2)*sp.spdiags(kappa*N12, 0, 2*Nz, 2*Nz)),(n12aux + dt/(epsilon**2)*r*B12))
    n1 = n12[:Nz]
    n2 = n12[Nz:]
    
    #### Scheme for RK 2001 moments based system
    M_RK = np.array([
        [-r0 + r0*RK[0]/K + gamma/2*(RK[2] + theta)**2 + gamma/2*sigmag**2 + m, -m, 0, 0],
        [-m, gamma/2*(RK[3] - theta)**2 + gamma/2*sigmag**2 - r0 + r0*RK[1]/K + m, 0, 0],
        [0, 0, sigmag**2*gamma + m*RK[1]/RK[0], -m*RK[1]/RK[0]], 
        [0, 0, -m*RK[0]/RK[1], sigmag**2*gamma + m*RK[0]/RK[1]],
        ])
    RK = scilin.solve(np.eye(4) + dt/sigmag**2*M_RK, RK + dt/sigmag**2*V_RK)
    
    return(n1, n2, RK)

####### Displays the comparison between a moment time series deduced from our scheme and RK 2001's one (size and mean)
def plot_comparison(T, y1, y2, y1_RK, y2_RK, ylim, yliminf, ylimsup, y_label, ylabel1, ylabel2, savepath, theta, add_plot):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('Time $t$ (log scale)', fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20) 
    ax.plot(T, y1_RK, color="darkblue", label = ylabel1 + ' [RK]')
    ax.plot(T, y2_RK, color="navy", label = ylabel2 + ' [RK]')
    ax.plot(T, y1, color="goldenrod", label = ylabel1)
    ax.plot(T, y2, color="darkgoldenrod", label = ylabel2)
    
    if add_plot:
        ax.set_ylim(yliminf, ylimsup)
        ax.plot(T, theta*np.ones(np.size(T)), linestyle='dashed', alpha=0.5, color='black')
        ax.plot(T, -theta*np.ones(np.size(T)), linestyle='dashed', alpha=0.5, color='black')

    plt.xscale('log')
    plt.legend(loc=0, fontsize=20)
    plt.savefig(savepath)
    plt.close()

####### Displays the moment time series deduced from our scheme (variance and skewness)

def plot_no_comp(T, y1, y2, y_RK, y_label, ylabel_RK, ylabel1, ylabel2, savepath, ylogscale):
    fig = plt.figure(figsize = (9,6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top = 0.85)
    ax.set_xlabel('Time $t$ (log scale)', fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)    
    ax.plot(T, y1, color = "goldenrod", label = ylabel1)
    ax.plot(T, y2, color = "darkgoldenrod", label = ylabel2)
    ax.plot(T, y_RK, color="navy", linestyle = '--', label = ylabel_RK)
    if ylogscale:
        plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc=0, fontsize=20)
    plt.savefig(savepath)
    plt.close()

######## Function that runs the whole scheme, given the initial distributions n1 and n2 and the initial moments (RK)
######## Produces final time series comparison between the moments generated by the two models and save them 
def run_model(n1, n2, RK, parameters, parameters_RK, T, Nt, dt, zmax, z, Nz, dz, title, subtitle, workdir, subworkdir):
    epsilon, r, kappa, g, m, theta = parameters
    sigmag, rstar, Kstar, gamma, m, theta1, theta2 = parameters_RK
    
    moments_1 = np.zeros((4,np.size(T)))
    moments_2 = np.zeros((4,np.size(T)))
    moments_RK = np.zeros((4,np.size(T)))
    
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
    V_RK  = np.array([0, 0, -sigmag**2*gamma*theta, sigmag**2*gamma*theta])

    for t in range(Nt):
        moments_1[:, t] = moments(n1, z, dz)
        moments_2[:, t] = moments(n2, z, dz)
        moments_RK[:, t] = RK
        n1, n2, RK = update(n1, n2, RK, parameters, parameters_RK, M_semi_implicit_scheme, V_RK, dt, dz, Nz, zmax)
    
    
    np.save(subworkdir +'/n1.npy', n1)
    np.save(subworkdir +'/n2.npy', n2)
    np.save(subworkdir +'/moments_1', moments_1)
    np.save(subworkdir +'/moments_2', moments_2)
    np.save(subworkdir +'/moments_RK', moments_RK)


    plot_comparison(T, moments_1[0, :], moments_2[0, :], moments_RK[0,:], moments_RK[1,:], False, 0, 0, 'Size of subpopulation', '$N_1$', '$N_2$', subworkdir + '/' + title + '_sizes.png', theta, add_plot=False)
    plot_comparison(T, moments_1[1, :], moments_2[1, :], moments_RK[2,:], moments_RK[3,:], True, -5/4*theta, 5/4*theta, 'Local mean traits', '$\overline{z}_1$', '$\overline{z}_2$', subworkdir + '/' + title + '_mean_traits.png', theta, add_plot=True)
    plot_no_comp(T, moments_1[2, :], moments_2[2, :], sigmag**2*np.ones(Nt), 'Variance', 'Fixed variance [RK]', 'Local variance in trait 1', 'Local variance in trait 2', subworkdir + '/' + title + "variance.png", ylogscale = (epsilon == 0.05))
    plot_no_comp(T, moments_1[3, :], moments_2[3, :], np.zeros(Nt), 'Skewness', 'Skew null [RK]', 'Skew 1', 'Skew 2', subworkdir + '/' + title + "skew.png", ylogscale = False)
