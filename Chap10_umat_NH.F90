!
! FILE:
!   umat_NH.F90
! AUTHOR:
!   Jorgen Bergstrom, Ph.D.
! CONTENTS:
!   Abaqus UMAT subroutine for the Neo-Hookean (NH) model.
!   The subroutine is an example and not is not designed to
!   be a commercial quality implementation.
! USAGE:
!   2 material constants:
!      1: mu    (shear modulus)
!      2: kappa (bulk modulus)
!
! COMMENTS:
!
!     This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".

subroutine umatErr(str)
    implicit none
    character(LEN=*), intent(in) :: str
    print '(a,a)', "UMAT Error:    ", trim(str)
    stop 3
end subroutine umatErr

subroutine umat(strVec, statev, ddsdde, &
        energyElast, energyPlast, energyVisc, &
        rpl, ddsddt, drplde, drpldt, stran, dstran, time, dtime, &
        temp, dtemp, predef, dpred, cmname, ndi, nshr, ntens, &
        nstatev, inProps, nrInProps, coords, drot, pnewdt, celent, &
        dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc)
    implicit none
    integer, intent (in) :: ndi, nshr, ntens, nstatev, nrInProps, &
        noel, npt, layer, kspt, kstep, kinc
    character(len=8), intent(in) :: cmname
    real, intent(inout) :: strVec(ntens), statev(nstatev)
    real, intent(inout) :: energyElast, energyPlast, energyVisc
    real, intent(out) :: ddsdde(ntens,ntens), rpl, &
        ddsddt(ntens), drplde(ntens), drpldt
    real, intent(in) :: stran(ntens), dstran(ntens), time(2), dtime, &
        temp, dtemp, predef(1), dpred(1)
    real, intent(in) :: inProps(nrInProps), coords(3), drot(3,3)
    real, intent(out) :: pnewdt
    real, intent(in) :: celent, dfgrd0(3,3)
    real, intent(inout) :: dfgrd1(3,3)

    ! local variables
    real :: J, a1, a2, kk, F(3,3), b(3,3), bs(3,3), Stress(3,3), devbs(3,3)
    real :: tmp, Eye(3,3), mu, kappa, I1s

    ! setup variables
    if (nrInProps /= 2) call umatErr("Wrong number of input parameters")
    mu = inProps(1)
    kappa = inProps(2)
    if (mu < 0) call umatErr("Parameter mu has to be positive")
    if (kappa < 0) call umatErr("Parameter kappa has to be positive")
    if (ndi /= 3 .or. nshr /= 3) call umatErr("This umat only works for 3D elements")
    pnewdt=2.0

    ! take the time-step
    F = dfgrd1
    J = F(1,1) * (F(2,2)*F(3,3) - F(2,3)*F(3,2)) + &
        F(1,2) * (F(2,3)*F(3,1) - F(2,1)*F(3,3)) + &
        F(1,3) * (F(2,1)*F(3,2) - F(2,2)*F(3,1))

    b(1,1) = F(1,1)*F(1,1) + F(1,2)*F(1,2) + F(1,3)*F(1,3)
    b(1,2) = F(1,1)*F(2,1) + F(1,2)*F(2,2) + F(1,3)*F(2,3)
    b(1,3) = F(1,1)*F(3,1) + F(1,2)*F(3,2) + F(1,3)*F(3,3)
    b(2,1) = b(1,2)
    b(2,2) = F(2,1)*F(2,1) + F(2,2)*F(2,2) + F(2,3)*F(2,3)
    b(2,3) = F(2,1)*F(3,1) + F(2,2)*F(3,2) + F(2,3)*F(3,3)
    b(3,1) = b(1,3)
    b(3,2) = b(2,3)
    b(3,3) = F(3,1)*F(3,1) + F(3,2)*F(3,2) + F(3,3)*F(3,3)

    bs = J**(-2.0/3.0) * b

    tmp = bs(1,1) + bs(2,2) + bs(3,3)
    devbs = bs
    devbs(1,1) = bs(1,1) - tmp/3.0
    devbs(2,2) = bs(2,2) - tmp/3.0
    devbs(3,3) = bs(3,3) - tmp/3.0

    Eye = 0.0
    Eye(1,1) = 1.0
    Eye(2,2) = 1.0
    Eye(3,3) = 1.0

    Stress = mu/J * devbs + kappa*(J-1.0) * Eye

    I1s = bs(1,1) + bs(2,2) + bs(3,3)
    energyElast = 0.5*mu*(I1s - 3.0) + 0.5*kappa*(J-1.0)**2.0

    ! calculate the Jacobian
    a1 = mu / (9.0 * J)
    a2 = mu / (6.0 * J)
    kk = kappa * J

    ddsdde(1,1) = a1 * ( 8*bs(1,1) + 2*bs(2,2) + 2*bs(3,3)) + kk
    ddsdde(2,1) = a1 * (-4*bs(1,1) - 4*bs(2,2) + 2*bs(3,3)) + kk
    ddsdde(3,1) = a1 * (-4*bs(1,1) + 2*bs(2,2) - 4*bs(3,3)) + kk
    ddsdde(4,1) = a1 * 3*bs(1,2)
    ddsdde(5,1) = a1 * 3*bs(1,3)
    ddsdde(6,1) = a1 * (-6)*bs(2,3)

    ddsdde(1,2) = a1 * (-4*bs(1,1) - 4*bs(2,2) + 2*bs(3,3)) + kk
    ddsdde(2,2) = a1 * ( 2*bs(1,1) + 8*bs(2,2) + 2*bs(3,3)) + kk
    ddsdde(3,2) = a1 * ( 2*bs(1,1) - 4*bs(2,2) - 4*bs(3,3)) + kk
    ddsdde(4,2) = a1 * 3*bs(1,2)
    ddsdde(5,2) = a1 * (-6)*bs(1,3)
    ddsdde(6,2) = a1 * 3*bs(2,3)

    ddsdde(1,3) = a1 * (-4*bs(1,1) + 2*bs(2,2) - 4*bs(3,3)) + kk
    ddsdde(2,3) = a1 * ( 2*bs(1,1) - 4*bs(2,2) - 4*bs(3,3)) + kk
    ddsdde(3,3) = a1 * ( 2*bs(1,1) + 2*bs(2,2) + 8*bs(3,3)) + kk
    ddsdde(4,3) = a1 * (-6)*bs(1,2)
    ddsdde(5,3) = a1 * 3*bs(1,3)
    ddsdde(6,3) = a1 * 3*bs(2,3)

    ddsdde(1,4) = a2 * 2*bs(1,2)
    ddsdde(2,4) = a2 * 2*bs(1,2)
    ddsdde(3,4) = a2 * (-4)*bs(1,2)
    ddsdde(4,4) = a2 * 3*(bs(1,1) + bs(2,2))
    ddsdde(5,4) = a2 * 3*bs(2,3)
    ddsdde(6,4) = a2 * 3*bs(1,3)

    ddsdde(1,5) = a2 * 2*bs(1,3)
    ddsdde(2,5) = a2 * (-4)*bs(1,3)
    ddsdde(3,5) = a2 * 2*bs(1,3)
    ddsdde(4,5) = a2 * 3*bs(2,3)
    ddsdde(5,5) = a2 * 3*(bs(1,1) + bs(3,3))
    ddsdde(6,5) = a2 * 3*bs(1,2)

    ddsdde(1,6) = a2 * (-4)*bs(2,3)
    ddsdde(2,6) = a2 * 2*bs(2,3)
    ddsdde(3,6) = a2 * 2*bs(2,3)
    ddsdde(4,6) = a2 * 3*bs(1,3)
    ddsdde(5,6) = a2 * 3*bs(1,2)
    ddsdde(6,6) = a2 * 3*(bs(2,2) + bs(3,3))

    ! return variables:
    strVec(1) = Stress(1,1)
    strVec(2) = Stress(2,2)
    strVec(3) = Stress(3,3)
    strVec(4) = Stress(1,2)
    strVec(5) = Stress(1,3)
    strVec(6) = Stress(2,3)
    energyPlast = 0.0
    energyVisc = 0.0
    rpl = 0.0
    ddsddt(1) = 0.0
    drplde = 0.0
    drpldt = 0.0
end subroutine umat
