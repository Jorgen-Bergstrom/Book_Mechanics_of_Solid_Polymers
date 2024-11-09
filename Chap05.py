# Author:
#    Jorgen Bergstrom, Ph.D.
# Comments:
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".

import sys
import math
import numpy as np
import scipy.optimize


def uniaxial_stress(model, trueStrainVec, params):
    """Compressible uniaxial loading. Returns true stress."""
    stress = np.zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = np.exp(trueStrainVec[i])
        calcS22Abs = lambda x: abs(model([lam1,x[0],x[0]],params)[1,1])
        # search for transverse stretch that gives S22=0
        lam2 = scipy.optimize.fmin(calcS22Abs, x0=1/np.sqrt(lam1),
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,lam2[0],lam2[0]], params)[0,0]
    return stress


def biaxial_stress(model, trueStrainVec, params):
    """Compressible biaxial loading. Returns true stress."""
    stress = np.zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = np.exp(trueStrainVec[i])
        calcS33Abs = lambda x: abs(model([lam1,lam1,x[0]],params)[2,2])
        # search for transverse stretch that gives S33=0
        lam3 = scipy.optimize.fmin(calcS33Abs, x0=1/np.sqrt(lam1),
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,lam1,lam3[0]], params)[0,0]
    return stress


def planar_stress(model, trueStrainVec, params):
    """Compressible planar loading. Returns true stress."""
    stress = np.zeros(len(trueStrainVec))
    for i in range(len(trueStrainVec)):
        lam1 = np.exp(trueStrainVec[i])
        calcS33Abs = lambda x: abs(model([lam1,1.0,x[0]],params)[2,2])
        # search for transverse stretch that gives S33=0
        lam3 = scipy.optimize.fmin(calcS33Abs, x0=1/np.sqrt(lam1),
                                   xtol=1e-9, ftol=1e-9, disp=False)
        stress[i] = model([lam1,1.0,lam3[0]], params)[0,0]
    return stress


def NH_3D(stretch, param):
    """Neo-Hookean. 3D loading specified by stretches.
       param[0]=mu, param[1]=kappa"""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.3) * np.dot(F, F.T)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    return param[0]/J * devbstar + param[1]*(J-1) * np.eye(3)


def MR_3D(stretch, param):
    """Mooney-Rivlin. 3D loading specified by stretches.
       param: [C10, C01, kappa]"""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    bstar2 = np.dot(bstar, bstar)
    I1s = np.trace(bstar)
    I2s = 0.5 * (I1s**2 - np.trace(bstar2))
    C10 = param[0]
    C01 = param[1]
    kappa = param[2]
    return 2/J*(C10+C01*I1s)*bstar - 2*C01/J*bstar2 + \
        (kappa*(J-1) - 2*I1s*C10/(3*J) - 4*I2s*C01/(3*J))*np.eye(3)


def Yeoh_3D(stretch, param):
    """Yeoh. 3D loading specified by stretches.
       param: [C10, C20, C30, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    I1s = np.trace(bstar)
    return 2/J*(param[0] + 2*param[1]*(I1s-3) + 3*param[2]*(I1s-3)**2)*devbstar \
        + param[3]*(J-1) * np.eye(3)


def invLangevin(x):
    EPS = sys.float_info.epsilon
    if type(x) == float or type(x) == np.float64: # x is a scalar
        if x >= 1-EPS: x = 1 - EPS
        if x <= -1+EPS: x = -1 + EPS
        if abs(x) < 0.839:
            return 1.31435 * math.tan(1.59*x) + 0.911249*x
        return 1.0 / (sign(x) - x)
    # x is an array
    x[x >= 1-EPS] = 1 - EPS
    x[x <= -1+EPS] = -1 + EPS
    res = np.zeros(len(x))
    index = abs(x) < 0.839
    res[index] = 1.31435 * np.tan(1.59*x[index]) + 0.911249*x[index]
    index = abs(x) >= 0.839
    res[index] = 1.0 / (np.sign(x[index]) - x[index])
    return res


def EC_3D(stretch, param):
    """Eight-Chain. 3D loading specified by stretches.
       param: [mu, lambdaL, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    lamChain = np.sqrt(np.trace(bstar)/3)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    return param[0]/(J*lamChain) * invLangevin(lamChain/param[1]) / \
        invLangevin(1/param[1]) * devbstar + param[2]*(J-1) * np.eye(3)


def Gent_3D(stretch, param):
    """Gent. 3D loading specified by stretches.
       param: [mu, Jm, kappa]. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    I1s = np.trace(bstar)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    return param[0]/ J / (1 - (I1s-3)/param[1]) * devbstar + \
        param[2]*(J-1) * np.eye(3)


def HS_3D(stretch, param):
    """Horgan-Saccomandi. 3D loading specified by stretches.
       param: mu, lamMax, kappa. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    bstar2 = np.dot(bstar, bstar)
    I1s = np.trace(bstar)
    I2s = 0.5 * (I1s**2 - np.trace(bstar2))
    mu = param[0]
    lamM = param[1]
    kappa = param[2]
    fac = mu * lamM**4 / J
    den = lamM**6 - lamM**4 * I1s + lamM**2 * I2s - 1
    return fac/den * ((lamM**2 - I1s)*bstar + bstar2 - (lamM**2*I1s-2*I2s)/3*np.eye(3)) \
        + kappa*(J-1) * np.eye(3)


def Ogden_3D(stretch, param):
    """Ogden model. 3D loading specified by stretches.
       param: [mu1, mu2, ..., alpha1, alpha2, kappa]. Returns true stress."""
    J = stretch[0] * stretch[1] * stretch[2]
    lam = J**(-1/3) * np.array(stretch)
    N = round((len(param)-1)/2)
    mu = param[0:N]
    alpha = param[N:2*N]
    kappa = param[-1]
    Stress = kappa*(J-1)*np.eye(3)
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
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    lam = J**(-1/3) * np.array(stretch)
    I1s = np.trace(bstar)
    fac1 = (1+(1+I1s**2-4*I1s)*delta**2 + (5*I1s-I1s**2-6)*delta**4) / \
        (1 - (I1s-3)*delta**2)**2
    stressC = Gc/J * fac1 * devbstar
    fac2 = -2*Ge/(J*beta)
    tmp = (lam[0]**(-beta) + lam[1]**(-beta) + lam[2]**(-beta)) / 3
    stressE = np.zeros((3,3))
    stressE[0,0] = fac2 * (lam[0]**(-beta) - tmp)
    stressE[1,1] = fac2 * (lam[1]**(-beta) - tmp)
    stressE[2,2] = fac2 * (lam[2]**(-beta) - tmp)
    stressV = kappa*(J-1)*np.eye(3)
    return stressC + stressE + stressV


def Knowles_3D(stretch, param):
    """Knowles. 3D loading specified by stretches.
       param: mu, n, b, kappa. Returns true stress."""
    L1 = stretch[0]
    L2 = stretch[1]
    L3 = stretch[2]
    F = np.array([[L1,0,0], [0,L2,0], [0,0,L3]])
    J = np.linalg.det(F)
    bstar = J**(-2.0/3.0) * np.dot(F, F.T)
    I1s = np.trace(bstar)
    devbstar = bstar - np.trace(bstar)/3 * np.eye(3)
    return param[0]/J * (1+param[2]/param[1]*(I1s-3))**(param[1]-1) * devbstar \
        + param[3]*(J-1) * np.eye(3)


def blatzko_3D(stretch, param):
    """Blatz-Ko. 3D loading specified by stretches.
       param[0]=mu"""
    F = np.array([[stretch[0],0,0], [0,stretch[1],0], [0,0,stretch[2]]])
    J = np.linalg.det(F)
    b = np.dot(F, F.T)
    b2 = np.dot(b, b)
    I1 = np.trace(b)
    I2 = 0.5 * (I1**2 - np.trace(b2))
    return param[0]/J**3.0 * (I1*b - b2 - (I2-J**3.0) * np.eye(3))


def hyperfoam_3D(stretch, param):
    """Hyperfoam model. 3D loading specified by stretches.
       param: [mu1, mu2, ..., alpha1, alpha2, ..., beta1, beta2, ...].
       Returns true stress."""
    J = stretch[0] * stretch[1] * stretch[2]
    lam = np.array(stretch)
    N = int(round(len(param)/3.0))
    mu = param[0:N]
    alpha = param[N:2*N]
    beta = param[2*N:3*N]
    Stress = np.zeros((3,3))
    for k in range(N):
        fac = 2.0 * mu[k] / (J * alpha[k])
        Stress[0,0] = Stress[0,0] + fac*(lam[0]**alpha[k] - J**(-alpha[k]*beta[k]))
        Stress[1,1] = Stress[1,1] + fac*(lam[1]**alpha[k] - J**(-alpha[k]*beta[k]))
        Stress[2,2] = Stress[2,2] + fac*(lam[2]**alpha[k] - J**(-alpha[k]*beta[k]))
    return Stress


if __name__ == '__main__':
        print("")
        print("Inverse Langevin:")
        print(f"  invL(0.3) = {invLangevin(0.3)}")
        x = np.array([0.2, 0.3])
        print(f"  invL([0.2, 0.3]) = {invLangevin(x)}")

        print("NH")
        strain = np.array([0.0, 0.1, 0.2])
        params = np.array([1.0, 100])
        res = uniaxial_stress(NH_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(NH_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(NH_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("MR")
        params = np.array([1.0, 0.1, 100])
        res = uniaxial_stress(MR_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(MR_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(MR_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Yeoh")
        params = np.array([1.0, -0.1, 0.01, 100])
        res = uniaxial_stress(Yeoh_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(Yeoh_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(Yeoh_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("EC")
        params = np.array([1.0, 2.9, 100])
        res = uniaxial_stress(EC_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(EC_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(EC_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Gent")
        params = np.array([1.0, 8.9, 100])
        res = uniaxial_stress(Gent_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(Gent_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(Gent_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("HS")
        params = np.array([1.0, 8.9, 100])
        res = uniaxial_stress(HS_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(HS_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(HS_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Ogden")
        params = np.array([1.0, 1.1, 0.4, 2.02, 100])
        res = uniaxial_stress(Ogden_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(Ogden_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(Ogden_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("ETube")
        params = np.array([1.0, 1.1, 0.4, 2.02, 100])
        res = uniaxial_stress(ETube_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(ETube_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(ETube_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Knowles")
        params = np.array([1.0, 1.1, 0.4, 100])
        res = uniaxial_stress(Knowles_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(Knowles_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(Knowles_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Blatz-Ko")
        params = np.array([1.0])
        res = uniaxial_stress(blatzko_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(blatzko_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(blatzko_3D, strain, params)
        print(f"  planar stress:   {res}")

        print("Hyperfoam")
        params = np.array([1.0, 1.1, 0.3, 0.5, 2.0, 0.3])
        res = uniaxial_stress(hyperfoam_3D, strain, params)
        print(f"  uniaxial stress: {res}")
        res = biaxial_stress(hyperfoam_3D, strain, params)
        print(f"  biaxial stress:  {res}")
        res = planar_stress(hyperfoam_3D, strain, params)
        print(f"  planar stress:   {res}")
