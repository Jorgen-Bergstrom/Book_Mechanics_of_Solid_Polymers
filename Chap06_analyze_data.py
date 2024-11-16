# Author:
#    Jorgen Bergstrom, Ph.D.
# Comments:
#    This file is distributed with my book:
#    "Mechanics of Solid Polymers - Theory and Computational Modeling".

import math
import numpy as np
import matplotlib.pyplot as plt

### Input
freq_Hz = 1.4
strain_mean = 0.0   # CANNOT HAVE A MEAN STRAIN
strain_amp = 0.05
stress_mean = 0.0
nr_cycles = 10
nr_dpts_per_cycle = 144
Ep = 2.0
Epp = 0.5

### Create virtual time-strain-stress data
freq_rad = freq_Hz * 2.0 * math.pi
delta = math.atan(Epp / Ep)
time = np.linspace(0.0, 1.0 * 2.0*math.pi*nr_cycles / freq_rad, nr_cycles * nr_dpts_per_cycle)
strain = strain_mean + strain_amp * np.sin(time * freq_rad) + \
    0.05 * strain_amp * np.random.rand(len(time))
stress = stress_mean + strain_amp * Ep * np.sin(time * freq_rad) + \
    strain_amp * Epp * np.cos(time * freq_rad) + \
    0.05 * Epp * np.random.rand(len(time))

### Extract the storage and loss moduli
hanning_window = np.hanning(len(time))
fft_strain = np.fft.fft(strain * hanning_window)
fft_stress = np.fft.fft(stress * hanning_window)
freq = np.fft.fftfreq(len(strain), time[1]-time[0])

fft_strain_mag = np.abs(fft_strain)
i = np.argmax(fft_strain_mag)
Gstar = fft_stress[i] / fft_strain[i]
Ep_calc = Gstar.real
Epp_calc = abs(Gstar.imag)
print(f"Ep_calc = {Ep_calc}")
print(f"Epp_calc = {Epp_calc}")

plt.plot(strain, stress)
plt.xlabel('Strain')
plt.ylabel('Stress')
plt.grid()
plt.show()
