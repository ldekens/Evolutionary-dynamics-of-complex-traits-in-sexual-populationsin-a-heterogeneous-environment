import numpy as np
import tools_fig6 as tools


######## Working parameters
parameters = 0.05, 1, 1, 1

######## Simulation's title

title = "fig7_top"

######## Creating directory

workdir = title
tools.create_directory(workdir)

######## Time discretization

Tmax, Nt = 300, 5000
T, dt = np.linspace(0, Tmax, Nt, retstep=True)

######## Trait space discretization

zmax, Nz = 1.5, 601
z, dz = np.linspace(-zmax, zmax, Nz, retstep=True)

######## Initial state for all simulations : dimorphic

epsilon, theta = parameters[0], parameters[3]
n1initial = .9*tools.Gauss(-theta, epsilon, z)
n2initial = 1.*tools.Gauss(theta, epsilon, z)

######## Run model

#tools.run_outcomes(n1initial, n2initial, parameters, T, Nt, dt, zmax, z, Nz, dz, title, workdir)
outcomes = np.load("fig7_top.npy")
tools.plot_summary_outcomes(outcomes, title)