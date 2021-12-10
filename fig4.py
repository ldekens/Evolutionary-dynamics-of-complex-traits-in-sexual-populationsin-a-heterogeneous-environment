#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:47:43 2021

@author: dekens
"""
import numpy as np
import matplotlib as ml
from matplotlib import cm
import matplotlib.pyplot as plt

##### Graphical settings
ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams.update({
    "text.usetex": True})
viridis = cm.get_cmap('viridis', 300)

##### Title
title = 'fig4'

##### Selection strengths (g) and migration rates (m)
g = np.arange(0, 3, 0.01)
m = np.arange(0, 3, 0.01)

##### Migration rate threshold given selection strength for the asymmetrical equilibria
m_thresh_asym = (5*g-1)/2.
m_thresh_asym[g>1] = 2*g[g>1]-2*np.sqrt(g[g>1]*(g[g>1]-1.))


##### Displays Figure 4 according to Proposition 4.1 and Proposition 4.2
fig = plt.figure(figsize = (10, 10))

plt.fill_between(g[g>=1], 0, m_thresh_asym[g>=1], color=viridis(150), alpha = 0.7, label='Asymmetrical equilibria')
plt.fill_between(g[g<=1], 0, m_thresh_asym[g<=1], color=viridis(200), alpha = 0.8, label='Coexistence of asym. and sym. eq.')
plt.fill_between(g[g<=1], m_thresh_asym[g<=1], 10, color=viridis(250), alpha=0.9, label="Symmetrical equilibrium")
plt.fill_between(g[g>=1], m_thresh_asym[g>=1], 10, color=viridis(0), label='No (fast) equilibrium')

plt.xlabel('Intensity of selection $(g)$', fontsize = 30)
plt.ylabel('Migration rate $(m)$', fontsize = 30)

plt.axis([0, max(g), 0, max(m)], 'equal')
plt.legend(fontsize = 20)
plt.savefig(title + '.png')
plt.savefig(title + '.eps')
plt.show()
plt.close()
