#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 18:06:56 2021

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
title = 'fig6_top'

##### Selection strengths (g) and migration rates (m)
g = np.arange(0, 3, 0.01)
m = np.arange(0, 3, 0.01)

##### Thresholds on migration rate given selection strength - summary of equilibrium analysis

m_thresh_asym = (5*g-1)/2.
m_thresh_asym[g>1] = 2*g[g>1]-2*np.sqrt(g[g>1]*(g[g>1]-1.))
m_thresh_sym_ext = (g-1)/2

##### Displays the top panel of Figure 6

fig = plt.figure(figsize = (10, 10))
plt.fill_between(g[g>=1], 2*g[g>=1]-2*np.sqrt(g[g>=1]*(g[g>=1]-1)), 10, color=viridis(0), label='Extinction')
plt.fill_between(g, m_thresh_sym_ext, m_thresh_asym, color=viridis(150), alpha=0.7, label='Specialist species (asym.)')
plt.fill_between(g[g>1], 0, m_thresh_sym_ext[g>1], color=viridis(150), alpha=0.7, hatch='//', label="Possible extinction, depending on initial state")
plt.fill_between(g[g<=1], m_thresh_asym[g<=1], 10, color=viridis(250), label="Generalist species (sym.)")

plt.xlabel('Intensity of selection $(g)$', fontsize=40)
plt.ylabel('Migration rate $(m)$', fontsize=40)
plt.yticks(fontsize=20)
plt.xticks(ticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=20)

plt.axis([0, max(g), 0, max(m)], 'equal')
plt.legend(fontsize = 25)
plt.savefig(title + '.png')
plt.savefig(title + '.eps')
plt.show()
plt.close()