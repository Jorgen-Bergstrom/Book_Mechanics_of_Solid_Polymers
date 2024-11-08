from Polymer_Mechanics_Chap05 import *
from Polymer_Mechanics_Chap09 import *


N = 100
timeVec = linspace(0, 10.0, N)
trueStrain = linspace(0, 0.2, N)
params = [2.0, 3.5, 500.0, 3.0, 0.05, -0.5, 0.5, 8.0, 0.01]
trueStress = uniaxial_stress_visco(BB_3D, timeVec, trueStrain, params)

plot(trueStrain, trueStress, 'r-', label='Python Calculation')
xlabel('True Strain')
ylabel('True Stress (MPa)')
grid('on')
savefig('BB_compressible_uniaxial.png', dpi=300)
show()

