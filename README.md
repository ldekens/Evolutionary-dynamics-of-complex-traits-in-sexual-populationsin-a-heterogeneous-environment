# Evolutionary-dynamics-of-complex-traits-in-sexual-populationsin-a-heterogeneous-environment
Codes for "Evolutionary dynamics of complex traits in sexual populations in a heterogeneous environment:  how normal?"

This repository contains the Python files that were used to produce the figures of the paper "Evolutionary dynamics of complex traits in sexual populations in a heterogeneous environment:  how normal?".
An open access version can be found at https://arxiv.org/abs/2012.10115.

The files tools_fig2.py and tools_fig6.py should be stored in the same directory as fig2.py, fig6_bottom.py, fig7_top.py and fig7_bottom.py.

Note that errors might occur when running the code on a machine or cluster that do not have a LaTeX programm installed. In that case, comment the occurences of:

ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    "text.usetex": True})
    
