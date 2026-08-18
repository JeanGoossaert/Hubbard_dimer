import numpy as np
from Exact import ExactChi11
from Exact_fxc import Exact_fxc_ab
from ExactKS import KS
from HF import RHF, UHF
from General_functions import ExpectationValue, Correlation_potential



"""
# (Not needed) residue calculation for poles of fxc
def Residues(omega, e, Omega_list, omega0):

    Residue_e = [0,0] ; Residue_h = [0,0]
    e_minus, e_plus = e

    for i,Omega in enumerate(Omega_list):

        ## 1. 1 pole in the upper half complex plane
        Residue_e[i] += 1 / (omega - e_minus + Omega) * 1 / (omega - e_minus + omega0)
        Residue_h[i] += 1 / (omega - e_plus - Omega) * 1 / (Omega + omega0)
        Residue_h[i] += 1 / (omega - e_plus - omega0) * 1 / (Omega + omega0)

        ## 2. 2 poles in the upper half complex plane

        # 2.1 e_minus and Omega
        f = 1 / (omega - e_minus - Omega)
        Residue_e[i] += f * (-1 / (omega - e_minus + omega0) + 1 / (Omega + omega0))

        # 2.2 e_minus and omega0
        f = 1 / (omega - e_minus - omega0)
        Residue_e[i] += f * (-1 / (omega - e_minus + Omega) + 1 / (Omega + omega0))

        # 2.3 Omega and omega0
        f = 1 / (Omega - omega0)
        Residue_h[i] += f * (-1 / (omega - e_plus - Omega) + 1 / (omega - e_plus - omega0))

        ## 3. 3 poles in the upper half complex plane
        f1 = 1 / (omega - e_minus - Omega)
        f2 = 1 / (omega - e_minus - omega0)
        f3 = 1 / (Omega - omega0)
        Residue_e[i] += f1 * f2 - f1 * f3 + f2 * f3

    return Residue_e, Residue_h



## Derivative of the residues with respect to omega, needed for the Z factor
def Residues_derivative(omega, e, Omega_list, omega0):

    dResidue_e = [0,0] ; dResidue_h = [0,0]
    e_minus, e_plus = e

    for i,Omega in enumerate(Omega_list):

        ## 1. 1 pole in the upper half complex plane
        dResidue_e[i] += -1 / (omega - e_minus + Omega)**2 * 1 / (omega - e_minus + omega0) + 1 / (omega - e_minus + Omega) * (-1) / (omega - e_minus + omega0)**2
        dResidue_h[i] += -1 / (omega - e_plus - Omega)**2 * 1 / (Omega + omega0)
        dResidue_h[i] += -1 / (omega - e_plus - omega0)**2 * 1 / (Omega + omega0)

        ## 2. 2 poles in the upper half complex plane

        # 2.1 e_minus and Omega
        f = 1 / (omega - e_minus - Omega)
        df = -1 / (omega - e_minus - Omega)**2
        dResidue_e[i] += df * (-1 / (omega - e_minus + omega0) + 1 / (Omega + omega0))
        dResidue_e[i] += f * 1 / (omega - e_minus + omega0)**2

        # 2.2 e_minus and omega0
        f = 1 / (omega - e_minus - omega0)
        df = -1 / (omega - e_minus - omega0)**2
        dResidue_e[i] += df * (-1 / (omega - e_minus + Omega) + 1 / (Omega + omega0))
        dResidue_e[i] += f * 1 / (omega - e_minus + Omega)**2

        # 2.3 Omega and omega0
        f = 1 / (Omega - omega0)
        dResidue_h[i] += f * (1 / (omega - e_plus - Omega)**2 - 1 / (omega - e_plus - omega0)**2)

        ## 3. 3 poles in the upper half complex plane
        f1 = 1 / (omega - e_minus - Omega) ; df1 = -1 / (omega - e_minus - Omega)**2
        f2 = 1 / (omega - e_minus - omega0) ; df2 = -1 / (omega - e_minus - omega0)**2
        f3 = 1 / (Omega - omega0)
        dResidue_e[i] += df1 * f2 + f1 * df2
        dResidue_e[i] += -df1 * f3
        dResidue_e[i] += df2 * f3

    return dResidue_e, dResidue_h
"""


def Z_G0Wtilde(G0, chi_exact, d, U, t):
    e, veff_list, dn = G0 ; gap = e[1] - e[0]
    A_list, Omega_list = chi_exact ; A1, A2 = A_list ; Omega1, Omega2 = Omega_list
    C1, C2, C3, omega0 = Exact_fxc_ab(d, U, t)

    # HOMO and LUMO have the same Z

    ## 1. G0W + static contribution
    dSigma_e_dE = -U*(U + C1) * (A1/(Omega1**2) + A2/(Omega2**2)) * (dn/2)**2
    dSigma_h_dE = -U*(U + C1) * (A1/(Omega1 + gap)**2 + A2/(Omega2 + gap)**2) * (1 - (dn/2)**2)

    ## 2. Quadratic contribution
    # Static correction to Sigma_x does not contribute to Z
    dSigma_e_dE += -U*C2 * (A1 + A2) * (dn/2)**2
    dSigma_h_dE += -U*C2 * (A1*Omega1**2/(Omega1 + gap)**2 + A2*Omega2**2/(Omega2 + gap)**2) * (1 - (dn/2)**2)

    ## 3. Pole contribution
    prefactor = -U*C3
    #dResidue_e, dResidue_h = Residues_derivative(e[0], e, Omega_list, omega0)
    #dSigma_e_dE += prefactor * (A1 * dResidue_e[0] + A2 * dResidue_e[1]) * (dn/2)**2
    #dSigma_h_dE += prefactor * (A1 * dResidue_h[0] + A2 * dResidue_h[1]) * (1-(dn/2)**2)

    def a(Omega):
        return -((Omega+omega0)**2 + Omega**2 + omega0**2)/(Omega**2 * omega0**2 *(Omega+omega0))
    def b(Omega):
        num = 2*(omega0*(gap+omega0)**2 - Omega*(gap+Omega)**2)
        denominator = (gap+Omega)**2 *(gap+omega0)**2 *(Omega+omega0)*(Omega-omega0)
        return num/denominator

    dSigma_e_dE += prefactor * (A1 * a(Omega1) + A2 * a(Omega2)) * (dn/2)**2
    dSigma_h_dE += prefactor * (A1 * b(Omega1) + A2 * b(Omega2)) * (1-(dn/2)**2)

    dSigma_dE = dSigma_e_dE + dSigma_h_dE
    return 1 / (1 - dSigma_dE)


def G0Wtilde_correction(G0, d, U, t):
    e, veff_list, dn = G0 ; gap = e[1] - e[0]
    vc_list = Correlation_potential(G0, d, U)
    vc_ij = [[vc_list[0], 0], [0, vc_list[1]]] # The correlation potential is diagonal in the site basis
    vc_HOMO = ExpectationValue(G0, 0, vc_ij, t) ; vc_LUMO = ExpectationValue(G0, 1, vc_ij, t)

    A_list, Omega_list = ExactChi11(d, U, t)
    C1, C2, C3, omega0 = Exact_fxc_ab(d, U, t)
    Z = Z_G0Wtilde(G0, (A_list, Omega_list), d, U, t)

    A1, A2 = A_list ; Omega1, Omega2 = Omega_list

    ## 1. G0W + static contribution
    Sigma_HOMO_e = U*(U + C1) * (A1/Omega1 + A2/Omega2) * (dn/2)**2
    Sigma_HOMO_h = U*(U + C1) * (A1/(Omega1 + gap) + A2/(Omega2 + gap)) * ((dn/2)**2 - 1)

    ## 2. Quadratic contribution
    # 2.1 static correction to Sigma_x
    prefactor = -U*C2 * (A1*Omega1 + A2*Omega2)
    n1 = (2 + dn) / 2 ; n2 = (2 - dn) / 2 ; n_ij = [[n1, 0], [0, n2]]
    Sigma_x_HOMO = ExpectationValue(G0, 0, n_ij, t) * prefactor
    Sigma_x_LUMO = ExpectationValue(G0, 1, n_ij, t) * prefactor

    # 2.2 correction to Sigma_c
    Sigma_HOMO_e += U*C2 * (A1*Omega1 + A2*Omega2) * (dn/2)**2
    Sigma_HOMO_h += U*C2 * (A1*Omega1**2/(Omega1 + gap) + A2*Omega2**2/(Omega2 + gap)) * ((dn/2)**2 - 1)

    ## 3. Pole contribution
    prefactor = -U*C3

    Sigma_HOMO_e += prefactor * 2*(A1/(Omega1*omega0)+A2/(Omega2*omega0)) * (dn/2)**2
    Sigma_HOMO_h += prefactor * (-2)*(A1*(gap+Omega1+omega0)/((gap+Omega1)*(gap+omega0)*(Omega1+omega0))
                                      +A2*(gap+Omega2+omega0)/((gap+Omega2)*(gap+omega0)*(Omega2+omega0))) * (1-(dn/2)**2)
    Sigma_LUMO_h = -Sigma_HOMO_e
    Sigma_LUMO_e = -Sigma_HOMO_h

    Sigma_HOMO = Sigma_HOMO_e + Sigma_HOMO_h + Sigma_x_HOMO
    Sigma_LUMO = Sigma_LUMO_e + Sigma_LUMO_h + Sigma_x_LUMO

    HOMO = e[0] + Z * (Sigma_HOMO - vc_HOMO)
    LUMO = e[1] + Z * (Sigma_LUMO - vc_LUMO)
    return HOMO, LUMO



def G0Wtilde_QP(d, U, t, StartingPoint='KS', ExactKSHomo=True):
    if StartingPoint == 'KS':
        G0 = KS(d, U, t, ExactKSHomo=ExactKSHomo)
    elif StartingPoint == 'RHF':
        G0 = RHF(d, U, t)
    return G0Wtilde_correction(G0, d, U, t)