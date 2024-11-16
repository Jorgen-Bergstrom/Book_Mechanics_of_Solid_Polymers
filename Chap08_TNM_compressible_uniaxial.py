# Author:
#    Jorgen Bergstrom, Ph.D.
# Comments:
#    This file is distributed with my book:
#    "Mechanics of Solid Polymers - Theory and Computational Modeling".

from Chap05 import *
from Chap09 import *

import numpy as np
import matplotlib.pyplot as plt

N = 50
timeVec = np.append(np.linspace(0,10.0,N), np.linspace(10.0,20.0,N)[1:])
trueStrain = np.append(np.linspace(0,0.2,N), np.linspace(0.2,0,N)[1:])

params = [290.0, 5.0, 2000.0, 7.0, 0.0, 9.5, \
        130.0, 50.0, 10.0, 24.0, 9.0, 8.0]

statev0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, params[6]] # FAv, FBv, muB

trueStress = uniaxial_stress_visco(TNM_3D, timeVec, trueStrain, params, statev0)

plt.plot(trueStrain, trueStress, 'r-', label='Python Calculation')
plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')
plt.grid('on')
plt.show()
