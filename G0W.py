import numpy as np
from Exact import ExactChi11
from ExactKS import KS
from HF import RHF, UHF
from General_functions import ExpectationValue, Correlation_potential


def Z_G0W(G0, chi_exact, U):
    e, veff_list, dn = G0 ; gap = e[1] - e[0]
    A_list, Omega_list = chi_exact ; A1, A2 = A_list ; Omega1, Omega2 = Omega_list
    
    dSigma_h_dE = - U**2 * (A1/(Omega1**2) + A2/(Omega2**2)) * (dn/2)**2
    dSigma_e_dE = - U**2 * (A1/(Omega1 + gap)**2 + A2/(Omega2 + gap)**2) * (1 - (dn/2)**2)
    dSigma_dE = dSigma_h_dE + dSigma_e_dE
    return 1 / (1 - dSigma_dE)



def G0W_correction(G0, d, U, t):
    e, veff_list, dn = G0 ; gap = e[1] - e[0]
    vc_list = Correlation_potential(G0, d, U)
    vc_ij = [[vc_list[0], 0], [0, vc_list[1]]] # The correlation potential is diagonal in the site basis
    vc_HOMO = ExpectationValue(G0, 0, vc_ij, t) ; vc_LUMO = ExpectationValue(G0, 1, vc_ij, t)

    A_list, Omega_list = ExactChi11(d, U, t)
    Z = Z_G0W(G0, (A_list, Omega_list), U)

    A1, A2 = A_list ; Omega1, Omega2 = Omega_list
    Sigma_HOMO_h = U**2 * (A1/Omega1 + A2/Omega2) * (dn/2)**2
    Sigma_HOMO_e = U**2 * (A1/(Omega1 + gap) + A2/(Omega2 + gap)) * ((dn/2)**2 - 1)

    Sigma_HOMO = Sigma_HOMO_h + Sigma_HOMO_e
    Sigma_LUMO = -Sigma_HOMO

    HOMO = e[0] + Z * (Sigma_HOMO - vc_HOMO)
    LUMO = e[1] + Z * (Sigma_LUMO - vc_LUMO)
    return HOMO, LUMO

def G0W_QP(d, U, t, StartingPoint='KS', ExactKSHomo=True):
    if StartingPoint == 'KS':
        G0 = KS(d, U, t, ExactKSHomo=ExactKSHomo)
    elif StartingPoint == 'RHF':
        G0 = RHF(d, U, t)
    return G0W_correction(G0, d, U, t)
