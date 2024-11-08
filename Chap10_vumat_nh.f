c     FILE:
c        vumat_nh.f
c     AUTHOR:
c        Jorgen Bergstrom
c     CONTENTS:
c        Abaqus VUMAT subroutine for the
c        Neo-Hookean (NH) model. The subroutine is an example
c        of how to write a VUMAT, and not is not designed to
c        be a commercial quality implementation.
c     USAGE:
c        2 material constants:
c           1: mu    (shear modulus)
c           2: kappa (bulk modulus)
c
c     |<- column 1 begins here
c     |
c     *User material, constants=2
c     **     mu,    kappa
c           5.0,    100.0
c     *Density
c     1000e-12
c
c     COMMENTS:
c
c     This file is distributed with my book: "Mechanics of Solid Polymers - Theory and Computational Modeling".
      subroutine vumat(nblock, ndi, nshr, nstatev, nfieldv, nprops,
     .     lanneal, stepTime, totTime, dt, cmname, coordMp,
     .     charLen, props, density, Dstrain, rSpinInc, temp0,
     .     U0, F0, field0, stressVec0, state0,
     .     intEne0, inelaEn0, temp1, U1,
     .     F1, field1, stressVec1, state1, intEne1, inelaEn1)
      implicit none
      integer nblock, ndi, nshr, nstatev, nfieldv, nprops, lanneal
      real stepTime, totTime, dt
      character*80 cmname
      real coordMp(nblock,*)
      real charLen, props(nprops), density(nblock),
     .     Dstrain(nblock,ndi+nshr), rSpinInc(nblock,nshr),
     .     temp0(nblock), U0(nblock,ndi+nshr),
     .     F0(nblock,ndi+nshr+nshr), field0(nblock,nfieldv),
     .     stressVec0(nblock,ndi+nshr), state0(nblock,nstatev),
     .     intEne0(nblock), inelaEn0(nblock), temp1(nblock),
     .     U1(nblock,ndi+nshr), F1(nblock,ndi+nshr+nshr),
     .     field1(nblock,nfieldv), stressVec1(nblock,ndi+nshr),
     .     state1(nblock,nstatev), intEne1(nblock), inelaEn1(nblock)

c     local variables
      real F(3,3), B(3,3), J, t1, t2, t3, mu, kappa
      integer i

      mu = props(1)
      kappa = props(2)

c     loop through all blocks
      do i = 1, nblock
c        setup F (upper diagonal part)
         F(1,1) = U1(i,1)
         F(2,2) = U1(i,2)
         F(3,3) = U1(i,3)
         F(1,2) = U1(i,4)
         if (nshr .eq. 1) then
            F(2,3) = 0.0
            F(1,3) = 0.0
         else
            F(2,3) = U1(i,5)
            F(1,3) = U1(i,6)
         end if

c        calculate J
         t1 = F(1,1) * (F(2,2)*F(3,3) - F(2,3)**2)
         t2 = F(1,2) * (F(2,3)*F(1,3) - F(1,2)*F(3,3))
         t3 = F(1,3) * (F(1,2)*F(2,3) - F(2,2)*F(1,3))
         J = t1 + t2 + t3
         t1 = J**(-2.0/3.0)

c        Bstar = J^(-2/3) F Ft   (upper diagonal part)
         B(1,1) = t1*(F(1,1)**2 + F(1,2)**2 + F(1,3)**2)
         B(2,2) = t1*(F(1,2)**2 + F(2,2)**2 + F(2,3)**2)
         B(3,3) = t1*(F(1,3)**2 + F(2,3)**2 + F(3,3)**2)
         B(1,2) = t1*(F(1,1)*F(1,2)+F(1,2)*F(2,2)+F(1,3)*F(2,3))
         B(2,3) = t1*(F(1,2)*F(1,3)+F(2,2)*F(2,3)+F(2,3)*F(3,3))
         B(1,3) = t1*(F(1,1)*F(1,3)+F(1,2)*F(2,3)+F(1,3)*F(3,3))

c        Stress = mu/J * Dev(Bstar) + kappa*log(J)/J * Eye
         t1 = (B(1,1) + B(2,2) + B(3,3)) / 3.0
         t2 = mu/J
         t3 = kappa * log(J) / J
         StressVec1(i,1) = t2 * (B(1,1) - t1) + t3
         StressVec1(i,2) = t2 * (B(2,2) - t1) + t3
         StressVec1(i,3) = t2 * (B(3,3) - t1) + t3
         StressVec1(i,4) = t2 * B(1,2)
         if (nshr .eq. 3) then
            StressVec1(i,5) = t2 * B(2,3)
            StressVec1(i,6) = t2 * B(1,3)
         end if

      end do
      return
      end
