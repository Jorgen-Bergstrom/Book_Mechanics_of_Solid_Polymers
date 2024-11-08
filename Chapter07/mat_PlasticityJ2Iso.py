from pylab import *
import scipy.optimize

# File:
#    mat_PlasticityJ2Iso.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".

def errfunc(dgamma, alpha0, epsP_vec, sigmaY_vec, E, stressTrial):
    sigmaY1 = interp(alpha0+dgamma, epsP_vec, sigmaY_vec)
    return abs(stressTrial - dgamma * E * sign(stressTrial) - sigmaY1)


def plasticity_J2iso(strain, params):
    """Uniaxial loading. [E, sigY0, sigY1, eps1, ...]"""
    N = len(params)
    E = params[0]
    sigmaY_vec = [params[1]]
    sigmaY_vec.extend(params[2::2])
    epsP_vec = [0.0]
    epsP_vec.extend(params[3::2])
    stress = zeros(len(strain))
    alpha0 = 0.0
    for i in range(1, len(strain)):
        stressTrial = stress[i-1] + E * (strain[i] - strain[i-1])
        sigmaY0 = interp(alpha0, epsP_vec, sigmaY_vec)
        fTrial = abs(stressTrial) - sigmaY0
        if fTrial < 0:
            stress[i] = stressTrial
        else:
            sigmaY0 = interp(alpha0, epsP_vec, sigmaY_vec)
            h0 = (interp(alpha0+1.0e-4,epsP_vec,sigmaY_vec) - sigmaY0) / 1.0e-4
            dgamma = fTrial / (E+h0)
            res = scipy.optimize.fmin(errfunc, x0=[dgamma], xtol=1e-9, \
                ftol=1e-9, maxfun=9999, full_output=1, disp=0, \
                args=(alpha0, epsP_vec, sigmaY_vec, E, stressTrial))
            dgamma = res[0][0]
            stress[i] = stressTrial - dgamma * E * sign(stressTrial)
            alpha0 = alpha0 + dgamma
    return stress


strain = linspace(0.0, 0.8, 10)
params = [50.0, 10.0, 15.0, 0.1]

stress = plasticity_J2iso(strain, params)

plot(strain, stress, 'b.-')
show()
