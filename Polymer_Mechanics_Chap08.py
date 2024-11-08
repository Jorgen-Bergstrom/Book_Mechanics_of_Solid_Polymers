from Polymer_Mechanics_Chap05 import *
import scipy.integrate

# File:
#    Polymer_Mechanics_Chap08.py
# Author:
#    Jorgen Bergstrom, Ph.D. (jorgen@polymerfem.com)
# Comments;
#    This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".


def ramp(x):
    return (x + abs(x)) / 2.0


def toVec(A):
    """Convert a 3x3 matrix to vector"""
    return array([A[0][0], A[1][1], A[2][2]])

def Dev(A):
    """Deviatoric part of a tensor"""
    return A - sum(A)/3.0


def Inv(A):
    """Inverse of a tensor"""
    return array([1.0, 1.0, 1.0]) / A


def pressure(A):
    """Pressure of a stress tensor"""
    return -sum(A) / 3.0


def uniaxial_stress_visco(model, timeVec, trueStrainVec, params):
    """Compressible uniaxial loading. Returns true stress."""
    stress = zeros(len(trueStrainVec))
    lam2_1 = 1.0
    FBv1 = array([1.0, 1.0, 1.0])
    for i in range(1, len(trueStrainVec)):
        print 'uniaxial_stress: i=', i, ' of ', len(trueStrainVec)
        time0 = timeVec[i-1]
        time1 = timeVec[i]
        lam1_0 = exp(trueStrainVec[i-1])
        lam1_1 = exp(trueStrainVec[i])
        lam2_0 = lam2_1
        F0 = array([lam1_0, lam2_0, lam2_0])
        F1 = array([lam1_1, lam2_1, lam2_1])
        FBv0 = FBv1.copy()
        calcS22Abs = lambda x: abs(model(F0, array([lam1_1,x,x]), \
                FBv0, time0, time1, params)[0][1])
        # search for transverse stretch that gives S22=0
        lam2_1 = scipy.optimize.fmin(calcS22Abs, x0=lam2_0, 
                                   xtol=1e-9, ftol=1e-9, disp=False)
        res = model(F0, array([lam1_1, lam2_1, lam2_1]), FBv0, time0, time1, params)
        stress[i] = res[0][0]
        FBv1 = res[1]
    return stress


def BB_timeDer_3D(Fv, t, params, time0, time1, F0, F1):
    """Returns FvDot"""
    mu, lamL, kappa, s, xi, C, tauBase, m, tauCut = params[:9]
    F = F0 + (t-time0) / (time1-time0) * (F1-F0)
    Fe = F / Fv
    Stress = toVec(EC_3D(Fe, [s*mu, lamL, kappa]))
    devStress = Stress - sum(Stress)/3
    tau = norm(devStress)
    lamCh = sqrt(sum(Fv*Fv)/3.0)
    lamFac = lamCh - 1.0 + xi
    gamDot = lamFac**C * (ramp(tau/tauBase-tauCut)**m)
    prefac = 0.0
    if tau > 0: prefac = gamDot / tau
    FeInv = array([1.0, 1.0, 1.0]) / Fe
    FvDot = prefac * (FeInv * devStress * F)
    return FvDot


def BB_3D(F0, F1, FBv0, time0, time1, params):
    """BB-model. 3D loading specified by principal stretches.
       params: [muA, lamL, kappa, s, xi, C, tauHat, m, tauCut].
       Returns: (true stress, FBv1)"""
    muA, lamL, kappa, s = params[:4]
    StressA = toVec(EC_3D(F1, [muA, lamL, kappa]))
    FBv1 = scipy.integrate.odeint(BB_timeDer_3D, FBv0, \
            [time0, time1], args=(params, time0, time1, F0, F1))[1]
    FBe1 = F1 / FBv1
    StressB = toVec(EC_3D(FBe1, [s*muA, lamL, kappa]))
    Stress = StressA + StressB
    return (Stress, FBv1)


def TNM_timeDer_3D(statev, t, params, time0, time1, F0, F1):
    """Returns statevDot"""
    muA, lamL, kappa = params[0:3]
    tauHatA, a, mA = params[3:6]
    muBi, muBf, beta = params[6:9]
    tauHatB, mB, muC = params[9:12]
    res = zeros(len(statev))
    F = F0 + (t-time0) / (time1-time0) * (F1-F0)

    # Network A: FAv
    Fv = statev[0:3]
    muB = statev[6]
    Fe = F / Fv
    Stress = toVec(EC_3D(Fe, [muA, lamL, kappa]))
    tau = norm(Dev(Stress)) + 1.0e-9
    gamDot = (tau / (tauHatA + a * ramp(pressure(Stress))))**mA
    res[0:3] = gamDot/tau * (Inv(Fe) * Dev(Stress) * F)
    res[6] = -beta * (statev[6] - muBf) * gamDot

    # Network B: FBv
    Fv = statev[3:6]
    muB = statev[6]
    Fe = F / Fv
    Stress = toVec(EC_3D(Fe, [muB, lamL, kappa]))
    tau = norm(Dev(Stress)) + 1.0e-9
    gamDot = (tau / (tauHatB + a * ramp(pressure(Stress))))**mB
    res[3:6] = gamDot/tau * (Inv(Fe) * Dev(Stress) * F)
    return res


def TNM_3D(F0, F1, statev0, time0, time1, params):
    """TN-model. 3D loading specified by principal stretches.
       params: [muA, lamL, kappa, tauHatA, a, mA, muBi, mBf,
                beta, tauHatB, mB, muC].
       Returns: (true stress, statev1)"""
    muA, lamL, kappa, s = params[0:4]
    muC = params[11]
    StressC = toVec(EC_3D(F1, [muC, lamL, kappa]))

    statev1 = scipy.integrate.odeint(TNM_timeDer_3D, statev0, \
            [time0, time1], args=(params, time0, time1, F0, F1))[1]
    FAv1 = statev1[0:3]
    FBv1 = statev1[3:6]
    muB = statev1[6]

    FAe1 = F1 / FAv1
    StressA = toVec(EC_3D(FAe1, [muA, lamL, kappa]))

    FBe1 = F1 / FBv1
    StressB = toVec(EC_3D(FBe1, [muB, lamL, kappa]))

    Stress = StressA + StressB + StressC
    return (Stress, statev1)

