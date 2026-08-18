import numpy as np
from Exact import ExactChi11
from ExactKS import KS


def Exact_fxc_ab(d, U, t):

    # Exact Chi
    [A2, A3], [Omega2, Omega3] = ExactChi11(d, U, t)
    A2 *= 2 ; A3 *= 2 # amplitude of the ab/ab element of Chi

    # Chi KS
    eKS, vslist, dn = KS(d, U, t)
    Omega0 = eKS[1]-eKS[0]
    A0 = 1-(dn/2)**2 # amplitude of the ab/ab element of Chi_KS

    X = A2*Omega2 + A3*Omega3
    Y = A2*Omega3 + A3*Omega2

    # Static term
    alpha = -Omega0/(2*A0) + (A2*Omega2**3 + A3*Omega3**3)/(2*X**2) - U
    
    # Quadratic term
    beta = 1/(2*A0*Omega0) -1/(2*X)

    # Pole term
    omega0 = np.sqrt(Omega2*Omega3*Y/X)
    gamma = A2*A3/(2*omega0*X**3) * (Omega2*Omega3)**3 * ((Omega2/Omega3)**2 /2 + (Omega3/Omega2)**2 /2 - 1)

    return alpha, beta, gamma, omega0