#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:57:49 2021

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
title = 'fig3'

##### Selection strengths (g) and migration rates (m)
g = np.arange(0, 3, 0.01)
m = np.arange(0, 3, 0.01)

##### Thresholds on migration rate given selection strength delimiting the regions determined by Proposition 3.1
m_thresh_1 = (1 - g[g<=1])/2
m_thresh_2 = 1 - g[g<=1]
m_thresh_3 = 2*g[g>=1] - 2*np.sqrt(g[g>=1]*(g[g>=1]-1))

##### Dsiplays Figure 3 according to the regions given by Proposition 3.1
plt.figure(figsize = (10, 10))
plt.fill_between(g[g<=1], 0, m_thresh_1, color=viridis(300), label='$z^* \in [0,\sqrt{(1-m)/g}-1[\cup]\sqrt{z_1},\sqrt{z_2}[$')
plt.fill_between(g[g<=1], m_thresh_1, m_thresh_2, color=viridis(250), label='$z^* \in [0,\max(\sqrt{(1-m)/g},\sqrt{z_2})[$')
plt.fill_between(g[g<=1], m_thresh_2, 6, color=viridis(200), label='$z^* \in [0,\sqrt{z_2}[$')
plt.fill_between(g[g>=1], 0, m_thresh_3, color=viridis(150), label='$z^* \in ]\sqrt{z_1},\sqrt{z_2}[$')
plt.fill_between(g[g>=1], m_thresh_3, 6, color=viridis(0), label='No fast equilibrium')

plt.xlabel('Intensity of selection $(g)$', fontsize = 30)
plt.ylabel('Migration rate $(m)$', fontsize = 30)

plt.axis([0, max(g), 0, max(m)], 'equal')
plt.legend(fontsize = 20)
plt.savefig(title + '.png')
plt.savefig(title + '.eps')
plt.show()
plt.close()