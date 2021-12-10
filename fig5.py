import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as ml

filetitle = "fig_5"

#### Parameters

g=1.5 ### selection strength
M = np.array([0.02,0.25,1,3.25,5]) ### migration rates

#### Trait space
zmax = 1.5
Nz = 2000
Z, dz=np.linspace(-zmax, zmax, Nz, retstep=True)

#### Auxiliary functions
def search_positive_roots(P): ## P a polynomial given by its list of coefficients
    roots = np.roots(P)
    roots = roots[np.real(np.isreal(roots))]
    roots = roots[roots>0]
    return(roots)

### Slow-dynamic function
def F(z, rho):
    return(2*g*((rho**2-1)/(rho**2+1) - z))

### Build the eventually multivalued graph of f
def curve_per_parameter(m): 
    rho_star, z_star = np.empty(3*Nz), np.empty(3*Nz)
    index_current = 0
    for k in range(Nz):
        z = Z[k]
        P = [1, 1/m - 1 - g/m*(z + 1)**2, -1/m + 1 + g/m*(z -1)**2, -1]
        roots = search_positive_roots(P)
        index_past = index_current
        index_current = index_past + np.size(roots)
        rho_star[index_past:index_current] = roots
        z_star[index_past:index_current] = z*np.ones(np.size(roots))
    z_star = z_star[:index_current]
    rho_star = rho_star[:index_current]
    return(z_star, list(map(F, z_star, rho_star)))             
    

#### Graphical output

## Graphical parameters
ml.rcParams['mathtext.fontset'] = 'stix'
ml.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({"text.usetex": True})
plt.figure(figsize=(9,9))
viridis_c = cm.get_cmap('viridis', 5)
plt.xlabel('$z*$', fontsize = 20)
plt.ylabel('$f\,(z*)$', fontsize = 20)
plt.ylim(-5, 5)
label_add =np.array(['$1+2m<g$','$1+2m=g$','$g<1+2m<5g$','$1+2m=5g$','$5g<1+2m$'])
for i in range(np.size(M)):
    m = M[-1-i ]
    z_star, F_z_star = curve_per_parameter(m)
    plt.scatter(z_star, F_z_star, s=0.5, color=viridis_c(i), label ='$m=$%4.2f'%m+', ' + label_add[-1-i])

plt.plot(Z, np.zeros(np.size(Z)), color='black', linewidth=1)
plt.legend(loc=0, markerscale= 8, fontsize=13)
plt.savefig(filetitle+".eps", format='eps')
plt.close()


