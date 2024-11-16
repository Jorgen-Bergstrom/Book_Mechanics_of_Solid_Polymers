# Author:
#    Jorgen Bergstrom, Ph.D.
# Comments:
#    This file is distributed with my book:
#    "Mechanics of Solid Polymers - Theory and Computational Modeling".

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d


def integrand(tau, tmax, timeVec, strainVec, params):
    E0 = params[0]
    g = params[1::2]
    tauG = params[2::2]
    ER = E0 * (1.0 - sum(g * (1 - np.exp((-tmax+tau)/tauG)  )))
    f = interp1d(timeVec, strainVec)
    edot = (f(tau+1e-5) - f(tau)) / 1e-5
    return ER * edot

def mat_LVE(timeVec, strainVec, params):
    """Linear viscoelasticity. Uniaxial loading."""
    stressVec = np.zeros(len(timeVec))
    for i in np.arange(len(timeVec)):
        tmax = timeVec[i]
        stressVec[i] = quad(integrand, 0, tmax, args=(tmax, timeVec, strainVec, params))[0]
    return stressVec


time = np.linspace(0, 2, 100)
strain = np.concatenate((np.linspace(0,0.5), np.linspace(0.5,0)))
stress = mat_LVE(time, strain, [1.0, 0.8, 0.1])
plt.plot(strain, stress, label='LVE')
plt.show()
