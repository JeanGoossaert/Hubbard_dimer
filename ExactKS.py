import numpy as np
from Exact import Exact_energies, Exact_DeltaN


    
def KS_dvs(dn, t):
    return 2*t*dn / np.sqrt(4 - dn**2)


def KS(d, U, t, ExactKSHomo=True):

    dn = Exact_DeltaN(d,U,t)
    dvs = KS_dvs(dn, t)
    
    if ExactKSHomo:
        [e0,e1,e2,e3], [e0_N1, e1_N1], [e0_N3, e1_N3], [a0,Norm0], [a2,Norm2], [a3,Norm3] = Exact_energies(d, U, t)
        homo = e0 - e0_N1
        vs_bar = homo - (-0.5)*np.sqrt((2*t)**2 + (dvs)**2)
    else:
        vs_bar = U/2
        
    vslist = [vs_bar-dvs/2, vs_bar+dvs/2]
    
    e0_KS = -0.5* np.sqrt((2*t)**2 + dvs**2) + vs_bar
    e1_KS = 0.5* np.sqrt((2*t)**2 + dvs**2) + vs_bar
    
    return [e0_KS, e1_KS], vslist, dn