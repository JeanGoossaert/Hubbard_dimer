import numpy as np
from Exact import Exact_energies, Non_interacting_density
from ExactKS import KS
from HF import RHF


def Hybrid(alpha, d, U, t, ExactKSHomo=True):
    eKS, vslist, dn = KS(d, U, t) ; dvxKS = dn*U/2
    eRHF, vRHFlist, dnRHF = RHF(d, U, t) ; dvxRHF = dnRHF*U/2

    # Replace a fraction alpha of the exchange part of vs by the exchange from RHF
    dvs = vslist[1] - vslist[0]
    dvs -= dvxKS ; dvs += alpha * dvxRHF + (1-alpha)*dvxKS
  
    if ExactKSHomo:
        [e0,e1,e2,e3], [e0_N1, e1_N1], [e0_N3, e1_N3], [a0,Norm0], [a2,Norm2], [a3,Norm3] = Exact_energies(d, U, t)
        homo = e0 - e0_N1
        vs_bar = homo - (-0.5)*np.sqrt((2*t)**2 + (dvs)**2)
    else:
        vs_bar = U/2

    vslist = [vs_bar-dvs/2, vs_bar+dvs/2]
    e0_KS = -0.5* np.sqrt((2*t)**2 + dvs**2) + vs_bar
    e1_KS = 0.5* np.sqrt((2*t)**2 + dvs**2) + vs_bar

    dn = Non_interacting_density(dvs, t)
    return [e0_KS, e1_KS], vslist, dn