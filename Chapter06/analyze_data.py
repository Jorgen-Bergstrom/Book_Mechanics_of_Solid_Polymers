from pylab import *

# File:
#    analyze_data.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".
#
#    Copyright (C) 2015  Jorgen Bergstrom


### Input
freq_Hz = 1.4
strain_mean = 0.0   # CANNOT HAVE A MEAN STRAIN
strain_amp = 0.05
stress_mean = 0.0
nr_cycles = 10.0
nr_dpts_per_cycle = 144.0
Ep = 2.0
Epp = 0.5

### Create virtual time-strain-stress data
freq_rad = freq_Hz * 2.0 * pi
delta = arctan(Epp / Ep)
time = linspace(0.0, 1.0 * 2.0*pi*nr_cycles / freq_rad, nr_cycles * nr_dpts_per_cycle)
strain = strain_mean + strain_amp * sin(time * freq_rad) + \
    0.05 * strain_amp * random(size(time))
stress = stress_mean + strain_amp * Ep * sin(time * freq_rad) + \
    strain_amp * Epp * cos(time * freq_rad) + \
    0.05 * Epp * random(size(time))

### Extract the storage and loss moduli
hanning_window = hanning(size(time))
fft_strain = fft(strain * hanning_window)
fft_stress = fft(stress * hanning_window)
freq = fftfreq(size(strain), time[1]-time[0])

fft_strain_mag = abs(fft_strain)
i = argmax(fft_strain_mag)
Gstar = fft_stress[i] / fft_strain[i]
Ep_calc = Gstar.real
Epp_calc = abs(Gstar.imag)

plot(strain, stress)
xlabel('Strain')
ylabel('Stress')
grid('on')
show()
