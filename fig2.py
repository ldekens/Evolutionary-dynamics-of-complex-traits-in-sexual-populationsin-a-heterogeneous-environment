import numpy as np
import tools_fig2 as tools


######## Working parameters

#### RK2001 parameters (from Fig 1. of Ronce and Kirpatrick (2001)) - the first line is for epsilon = 1, the second is for epsilon = 0.05
#parameters_RK = 1., 1., 2.5, 0.1, 0.1, 20., 27.
parameters_RK = 0.05, 1., 2.5, 0.1, 0.1, 20., 27.

#### Our parameters

parameters = tools.equivalence_RK2001_parameters(parameters_RK)
epsilon, r, kappa, g, m, theta = parameters

######## Simulation's title

title = "fig2"
subtitle = 'm =%4.2f'%m+'_g =%4.2f'%g+'_theta =%4.1f'%theta+'_epsilon = %4.2f'%epsilon

######## Creating directory

workdir = title
subworkdir = title+'/'+subtitle
tools.create_directory(workdir)
tools.create_directory(subworkdir)

######## Time discretization

Tmax, Nt = 1000, 200000
T, dt = np.linspace(1, Tmax, Nt, retstep=True)

######## Trait space discretization

zmax, Nz = 7, 851
z, dz = np.linspace(-zmax, zmax, Nz, retstep=True)

######## Initial state

n1initial = 1/kappa*0.9*tools.Gauss(-theta, epsilon, z)
n2initial = 1/kappa*1*tools.Gauss(theta, epsilon, z)
moments_initial_1 = tools.moments(n1initial, z, dz)
moments_initial_2 = tools.moments(n2initial, z, dz)
RK_initial = [moments_initial_1[0], moments_initial_2[0], moments_initial_1[1], moments_initial_2[1]]

######## Run model

tools.run_model(n1initial, n2initial, RK_initial, parameters, parameters_RK, T, Nt, dt, zmax, z, Nz, dz, title, subtitle, workdir, subworkdir)
'''moments_1 =np.load(subworkdir+'/moments_1.npy')
moments_2 =np.load(subworkdir+'/moments_2.npy')
moments_RK =np.load(subworkdir+'/moments_RK.npy')



tools.plot_comparison(T, moments_1[0, :], moments_2[0, :], moments_RK[0,:], moments_RK[1,:], False, 0, 0, 'Size of subpopulation', '$N_1$', '$N_2$', subworkdir + '/' + title + '_sizes.png', theta, add_plot=False)
tools.plot_comparison(T, moments_1[1, :], moments_2[1, :], moments_RK[2,:], moments_RK[3,:], True, -5/4*theta, 5/4*theta, 'Local mean traits', '$\overline{z}_1$', '$\overline{z}_2$', subworkdir + '/' + title + '_mean_traits.png', theta, add_plot=True)
   ''' 