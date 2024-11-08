from pylab import *

# File:
#    LVE_large_strain.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".


def NH(strain, params):
    """Neo-Hookean model. Incompressible uniaxial loading."""
    mu = params[0]
    lam = exp(strain)
    return mu * (lam*lam - 1/lam)

def LVE_uniax(time, strain, params):
    """Linear viscoelasticity. [mu, g1, tau1, g2, tau2, ...]"""
    stress = zeros(len(time))
    g = params[1::2]
    tau = params[2::2]
    stressV = zeros(len(g))
    stressH0 = NH(strain[0], params)
    for i in range(1, len(time)):
        stressH1 = NH(strain[i], params)
        dstressH = stressH1 - stressH0
        dt = time[i] - time[i-1]
        stress[i] = stressH1
        for j in range(len(g)):
            stressV[j] = exp(-dt/tau[j]) * stressV[j] + \
                g[j]*stressH0*(1 - exp(-dt/tau[j])) + \
                g[j]*dstressH/dt*(dt-tau[j]+tau[j]*exp(-dt/tau[j]))
            stress[i] = stress[i] - stressV[j]
        stressH0 = stressH1
    return stress

N = 100
time = linspace(0, 2, 2*N)
strain = concatenate((linspace(0,0.5,N), linspace(0.5,0,N)))

params = [1.0, 0.8, 0.1] # [mu, g, tau]
stress = LVE_uniax(time, strain, params)

plot(strain, stress, 'b.-', label='LVE')
show()
