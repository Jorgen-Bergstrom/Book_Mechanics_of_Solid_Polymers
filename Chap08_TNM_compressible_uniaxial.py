from Polymer_Mechanics_Chap05 import *
from Polymer_Mechanics_Chap09 import *

# File:
#    TNM_compressible_uniaxial.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".


N = 50
timeVec = append(linspace(0,10.0,N), linspace(10.0,20.0,N)[1:])
trueStrain = append(linspace(0,0.2,N), linspace(0.2,0,N)[1:])

params = [290.0, 5.0, 2000.0, 7.0, 0.0, 9.5, \
        130.0, 50.0, 10.0, 24.0, 9.0, 8.0]

# FAv, FBv, muB
statev0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, params[6]]

trueStress = uniaxial_stress_visco(TNM_3D, timeVec, trueStrain, params, statev0)

plot(trueStrain, trueStress, 'r-', label='Python Calculation')
xlabel('True Strain')
ylabel('True Stress (MPa)')
grid('on')
savefig('TNM_compressible_uniaxial.png', dpi=300)
show()

