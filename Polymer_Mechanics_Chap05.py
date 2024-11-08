from pylab import *
import scipy.optimize

# File:
#    Polymer_Mechanics_Chap05.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".
#
#    Copyright (C) 2015  Jorgen Bergstrom


def uniaxial_stress(model, trueStrainVec, params):
    """Compressible uniaxial loading. Returns true stress."""
    stress = zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = exp(trueStrainVec[i])
        calcS22Abs = lambda x: abs(model([lam1,x,x],params)[1,1])
        # search for transverse stretch that gives S22=0
        lam2 = scipy.optimize.fmin(calcS22Abs, x0=1/sqrt(lam1), 
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,lam2,lam2], params)[0,0]
    return stress


def biaxial_stress(model, trueStrainVec, params):
    """Compressible biaxial loading. Returns true stress."""
    stress = zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = exp(trueStrainVec[i])
        calcS33Abs = lambda x: abs(model([lam1,lam1,x],params)[2,2])
        # search for transverse stretch that gives S33=0
        lam3 = scipy.optimize.fmin(calcS33Abs, x0=1/sqrt(lam1), 
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,lam1,lam3], params)[0,0]
    return stress


def planar_stress(model, trueStrainVec, params):
    """Compressible planar loading. Returns true stress."""
    stress = zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = exp(trueStrainVec[i])
        calcS33Abs = lambda x: abs(model([lam1,1.0,x],params)[2,2])
        # search for transverse stretch that gives S33=0
        lam3 = scipy.optimize.fmin(calcS33Abs, x0=1/sqrt(lam1), 
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,1.0,lam3], params)[0,0]
    return stress


def NH_3D(stretch, param):
    """Neo-Hookean. 3D loading specified by stretches.
       param[0]=mu, param[1]=kappa"""
    F = array([[stretch[0],0,0], [0,stretch[1],0], [0,0,stretch[2]]])
    J = det(F)
    Fstar = J**(-1/3) * F
    bstar = dot(Fstar, Fstar.T)
    dev_bstar = bstar - trace(bstar)/3 * eye(3)
    return param[0]/J * dev_bstar + param[1]*(J-1) * eye(3)


def MR_3D(stretch, param):
    """Mooney-Rivlin. 3D loading specified by stretches.
       param: [C10, C01, kappa]"""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    bstar2 = dot(bstar, bstar)
    I1s = trace(bstar)
    I2s = 0.5 * (I1s**2 - trace(bstar2))
    C10 = param[0]
    C01 = param[1]
    kappa = param[2]
    return 2/J*(C10+C01*I1s)*bstar - 2*C01/J*bstar2 + \
        (kappa*(J-1) - 2*I1s*C10/(3*J) - 4*I2s*C01/(3*J))*eye(3)


def Yeoh_3D(stretch, param):
    """Yeoh. 3D loading specified by stretches.
       param: [C10, C20, C30, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    devbstar = bstar - trace(bstar)/3 * eye(3)
    I1s = trace(bstar)
    return 2/J*(param[0] + 2*param[1]*(I1s-3) + 3*param[2]*(I1s-3)**2)*devbstar \
        + param[3]*(J-1) * eye(3)


def invLangevin(x):
    EPS = spacing(1)
    if type(x) == float or type(x) == float64: # x is a scalar
        if x >= 1-EPS: x = 1 - EPS
        if x <= -1+EPS: x = -1 + EPS
        if abs(x) < 0.839:
            return 1.31435 * tan(1.59*x) + 0.911249*x
        return 1.0 / (sign(x) - x)
    # x is an array
    x[x >= 1-EPS] = 1 - EPS
    x[x <= -1+EPS] = -1 + EPS
    res = zeros(size(x))
    index = abs(x) < 0.839
    res[index] = 1.31435 * tan(1.59*x[index]) + 0.911249*x[index]
    index = abs(x) >= 0.839
    res[index] = 1.0 / (sign(x[index]) - x[index])
    return res


def EC_3D(stretch, param):
    """Eight-Chain. 3D loading specified by stretches.
       param: [mu, lambdaL, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    lamChain = sqrt(trace(bstar)/3)
    devbstar = bstar - trace(bstar)/3 * eye(3)
    return param[0]/(J*lamChain) * invLangevin(lamChain/param[1]) / \
        invLangevin(1/param[1]) * devbstar + param[2]*(J-1) * eye(3)


def Gent_3D(stretch, param):
    """Gent. 3D loading specified by stretches.
       param: [mu, Jm, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    I1s = trace(bstar)
    devbstar = bstar - trace(bstar)/3 * eye(3)
    return param[0]/ J / (1 - (I1s-3)/param[1]) * devbstar + \
        param[2]*(J-1) * eye(3)


def HS_3D(stretch, param):
    """Horgan-Saccomandi. 3D loading specified by stretches.
       param: mu, lamMax, kappa. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    bstar2 = dot(bstar, bstar)
    I1s = trace(bstar)
    I2s = 0.5 * (I1s**2 - trace(bstar2))
    mu = param[0]
    lamM = param[1]
    kappa = param[2]
    fac = mu * lamM**4 / J
    den = lamM**6 - lamM**4 * I1s + lamM**2 * I2s - 1
    return fac/den * ((lamM**2 - I1s)*bstar + bstar2 - (lamM**2*I1s-2*I2s)/3*eye(3)) \
        + kappa*(J-1) * eye(3)


def Ogden_3D(stretch, param):
    """Ogden model. 3D loading specified by stretches.
       param: [mu1, mu2, ..., alpha1, alpha2, kappa]. Returns true stress."""
    J = stretch[0] * stretch[1] * stretch[2]
    lam = J**(-1/3) * stretch
    N = round((len(param)-1)/2)
    mu = param[0:N]
    alpha = param[N:2*N]
    kappa = param[-1]
    Stress = kappa*(J-1)*eye(3)
    for i in range(N):
        fac = (2/J) * mu[i] / alpha[i]
        tmp = (lam[0]**alpha[i] + lam[1]**alpha[i] + lam[2]**alpha[i]) / 3
        Stress[0,0] = Stress[0,0] + fac * (lam[0]**alpha[i] - tmp)
        Stress[1,1] = Stress[1,1] + fac * (lam[1]**alpha[i] - tmp)
        Stress[2,2] = Stress[2,2] + fac * (lam[2]**alpha[i] - tmp)
    return Stress


def ETube_3D(stretch, param):
    """Extended Tube model. 3D loading specified by stretches.
       Param: Ge, Gc, delta, beta, kappa"""
    Ge = param[0]
    Gc = param[1]
    delta = param[2]
    beta = param[3]
    kappa = param[4]
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    devbstar = bstar - trace(bstar)/3 * eye(3)
    lam = J**(-1/3) * array(stretch)
    I1s = trace(bstar)
    fac1 = (1+(1+I1s**2-4*I1s)*delta**2 + (5*I1s-I1s**2-6)*delta**4) / \
        (1 - (I1s-3)*delta**2)**2
    stressC = Gc/J * fac1 * devbstar
    fac2 = -2*Ge/(J*beta)
    tmp = (lam[0]**(-beta) + lam[1]**(-beta) + lam[2]**(-beta)) / 3
    stressE = zeros((3,3))
    stressE[0,0] = fac2 * (lam[0]**(-beta) - tmp)
    stressE[1,1] = fac2 * (lam[1]**(-beta) - tmp)
    stressE[2,2] = fac2 * (lam[2]**(-beta) - tmp)
    stressV = kappa*(J-1)*eye(3)
    return stressC + stressE + stressV


def Knowles_3D(stretch, param):
    """Knowles. 3D loading specified by stretches.
       param: mu, n, b, kappa. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = det(F)
    bstar = J**(-2.0/3.0) * dot(F, F.T)
    I1s = trace(bstar)
    devbstar = bstar - trace(bstar)/3 * eye(3)
    return param[0]/J * (1+param[2]/param[1]*(I1s-3))**(param[1]-1) * devbstar \
        + param[3]*(J-1) * eye(3)


def blatzko_3D(stretch, param):
    """Blatz-Ko. 3D loading specified by stretches.
       param[0]=mu"""
    F = array([[stretch[0],0,0], [0,stretch[1],0], [0,0,stretch[2]]])
    J = det(F)
    b = dot(F, F.T)
    b2 = dot(b, b)
    I1 = trace(b)
    I2 = 0.5 * (I1**2 - trace(b2))
    return param[0]/J**3.0 * (I1*b - b2 - (I2-J**3.0) * eye(3))


def hyperfoam_3D(stretch, param):
    """Hyperfoam model. 3D loading specified by stretches.
       param: [mu1, mu2, ..., alpha1, alpha2, ..., beta1, beta2, ...].
       Returns true stress."""
    J = stretch[0] * stretch[1] * stretch[2]
    lam = array(stretch)
    N = int(round(len(param)/3.0))
    mu = param[0:N]
    alpha = param[N:2*N]
    beta = param[2*N:3*N]
    Stress = zeros((3,3))
    for k in range(N):
        fac = 2.0 * mu[k] / (J * alpha[k])
        Stress[0,0] = Stress[0,0] + fac*(lam[0]**alpha[k] - J**(-alpha[k]*beta[k]))
        Stress[1,1] = Stress[1,1] + fac*(lam[1]**alpha[k] - J**(-alpha[k]*beta[k]))
        Stress[2,2] = Stress[2,2] + fac*(lam[2]**alpha[k] - J**(-alpha[k]*beta[k]))
    return Stress
