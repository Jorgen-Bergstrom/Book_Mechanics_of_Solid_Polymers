# Author:
#    Jorgen Bergstrom, Ph.D.
# Comments:
#    This file is distributed with my book:
#    "Mechanics of Solid Polymers - Theory and Computational Modeling".

from Chap05 import *
from Chap09 import *

import numpy as np
import matplotlib.pyplot as plt

N = 100
timeVec = np.linspace(0, 10.0, N)
trueStrain = np.linspace(0, 0.2, N)
params = [2.0, 3.5, 500.0, 3.0, 0.05, -0.5, 0.5, 8.0, 0.01]
statev0 = [1.0, 1.0, 1.0]
trueStress = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, params, statev0)

plt.plot(trueStrain, trueStress, 'r-', label='Python Calculation')
plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')
plt.grid()
plt.show()
