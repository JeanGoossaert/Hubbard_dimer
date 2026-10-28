import numpy as np
from ExactKS import KS
from HF import RHF, UHF
from General_functions import *


## In the spin-restricted case, the RPA pole and amplitude have an simple analytical expression in the Hubbard dimer
def Chi_RPA(veff_list, U, t, eQP=None):
    [veff0, veff1] = veff_list
    dveff = veff1 - veff0 ; D2 = (dveff)**2
    Omega_RPA = np.sqrt(4*t**2 * (1 + D2) + 4*t*U / np.sqrt(1 + D2))
    A_RPA = t / (Omega_RPA * np.sqrt(1 + D2))

    # In the case of evGW, the poles are updated with the QP energies
    if eQP != None:
        D2 = (eQP[1]-eQP[0])**2 - (2*t)**2
        Omega_RPA = np.sqrt(4*t**2 * (1 + D2) + 4*t*U / np.sqrt(1 + D2))
    
    return A_RPA, Omega_RPA


## In general, the RPA poles and amplitudes can be computed from the Casida equation
# The next two functions are not used in this script but can be used to generalize G0W0 to bigger Hubbard models
def Chi_RPA_Casida(G0, U, t, SpinUnrestricted=False):
    e, veff_list, dn = G0
    if SpinUnrestricted:
        v = Single_particle_unrestricted_eigenstates(veff_list, t)
    else:
        v = Single_particle_restricted_eigenstates(veff_list, t)
    L = len(e) ; L2 = (L//2)*(L//2) # number of transitions valence -> conduction
    A = np.zeros((L2, L2)) ; B = np.zeros((L2, L2))
    Transitions = [(n,m) for n in range(L//2) for m in range(L//2,L)]
    for t1,transi1 in enumerate(Transitions):
        n1_1, n1_2 = transi1
        A[t1,t1] += e[n1_2] - e[n1_1]
        for t2,transi2 in enumerate(Transitions):
            n2_1, n2_2 = transi2
            a = 0
            for i in range(2):
                a += 2*U*v[n1_1, i]*v[n1_2, i]*v[n2_1, i]*v[n2_2, i]
            A[t1,t2] += a ; B[t1,t2] += a
    
    M = np.zeros((2*L2, 2*L2))
    M[:L2,:L2] = A ; M[:L2,L2:] = B ; M[L2:,:L2] = -B ; M[L2:,L2:] = -A
    enerlist, eigenvect = np.linalg.eig(M)
    enerlist, eigenvect = SortEigenstates(enerlist, eigenvect)
    return eigenvect[:,L2:], enerlist[L2:]


def RPA_amplitudes(eigenvect, G0, t, i,j, SpinUnrestricted=False):

    e, veff_list, dn = G0
    if SpinUnrestricted:
        v = Single_particle_unrestricted_eigenstates(veff_list, t)
    else:
        v = Single_particle_restricted_eigenstates(veff_list, t)
    L = len(e) ; L2 = (L//2)*(L//2)

    Transitions = [(n,m) for n in range(L//2) for m in range(L//2,L)] ; Amplitudes = []    
    for n in range(L2): # loop over the RPA poles
        a_i = 0 ; a_j = 0
        
        X = eigenvect[:L2,n] ; Y = eigenvect[L2:,n]
        norm = np.dot(X, X) - np.dot(Y, Y)
        X /= np.sqrt(norm) ; Y /= np.sqrt(norm)
        
        v_in = X + Y ; v_out = v_in # * D

        for t,transi in enumerate(Transitions):
            n1, n2 = transi
            a_i += v_out[t] * v[i,n1] * v[i,n2]
            a_j += v_out[t] * v[j,n1] * v[j,n2]
        Amplitudes.append(a_i * a_j)

    if SpinUnrestricted:
        return np.array(Amplitudes)
    else:
        return np.array(Amplitudes) *2



def Z_G0W0(G0, chi_RPA, U):
    e, veff_list, dn = G0
    A_RPA, Omega_RPA = chi_RPA
    gap = e[1] - e[0]
    dSigma_h_dE = - U**2 * A_RPA/(Omega_RPA**2) * (dn/2)**2
    dSigma_e_dE = - U**2 * A_RPA/(Omega_RPA + gap)**2 * (1 - (dn/2)**2)
    dSigma_dE = dSigma_h_dE + dSigma_e_dE
    return 1 / (1 - dSigma_dE)

def Z_G0W0_spin_unrestricted(G0, chi_RPA, U, spin):
    e, veff_list, dn = G0
    [A_up, A_down], [Omega_up, Omega_down] = chi_RPA
    gap = e[1] - e[0]
    dSigma_h_dE = - U**2 * (A_up/(Omega_up**2) + A_down/(Omega_down**2)) * dn**2
    dSigma_e_dE = - U**2 * (A_up/(Omega_up + gap)**2 + A_down/(Omega_down + gap)**2) * (1 - dn**2)
    dSigma_dE = dSigma_h_dE + dSigma_e_dE
    return 1 / (1 - dSigma_dE)


def G0W0_correction(G0, d, U, t):
    e, veff_list, dn = G0 ; gap = e[1] - e[0]
    vc_list = Correlation_potential(G0, d, U)
    vc_ij = [[vc_list[0], 0], [0, vc_list[1]]] # The correlation potential is diagonal in the site basis
    vc_HOMO = ExpectationValue(G0, 0, vc_ij, t) ; vc_LUMO = ExpectationValue(G0, 1, vc_ij, t)

    A_RPA, Omega_RPA = Chi_RPA(veff_list, U, t)
    Z = Z_G0W0(G0, (A_RPA, Omega_RPA), U)

    Sigma_HOMO_h = U**2 * A_RPA/Omega_RPA * (dn/2)**2
    Sigma_HOMO_e = U**2 * A_RPA/(Omega_RPA + gap) * ((dn/2)**2 - 1)

    Sigma_HOMO = Sigma_HOMO_h + Sigma_HOMO_e
    Sigma_LUMO = -Sigma_HOMO

    HOMO = e[0] + Z * (Sigma_HOMO - vc_HOMO)
    LUMO = e[1] + Z * (Sigma_LUMO - vc_LUMO)
    return HOMO, LUMO


def G0W0_correction_spin_unrestricted(G0, d, U, t):

    HOMOs, LUMOs = [], []
    E, Veff_list, Dn = G0
    e_up, e_down = E[:2], E[2:]
    veff_up, veff_down = Veff_list[0], Veff_list[1]
    dn_up, dn_down = Dn[0], Dn[1]

    A_up, Omega_up = Chi_RPA(veff_up, U, t) ; A_up /= 2
    A_down, Omega_down = Chi_RPA(veff_down, U, t) ; A_down /= 2
    
    for spin in range(2):
        e, veff_list, dn = [e_up, e_down][spin], [veff_up, veff_down][spin], [dn_up, dn_down][spin]
        G0_spin = [e, veff_list, dn] ; dn_bar = Dn[(spin+1)%2]
        gap = e[1] - e[0]
        vc_list = Correlation_potential(G0_spin, d, U, SpinUnrestricted=True, dn_bar=dn_bar)
        vc_ij = [[vc_list[0], 0], [0, vc_list[1]]] # The correlation potential is diagonal in the site basis
        vc_HOMO = ExpectationValue(G0_spin, 0, vc_ij, t) ; vc_LUMO = ExpectationValue(G0_spin, 1, vc_ij, t)

        Z = Z_G0W0_spin_unrestricted(G0_spin, ([A_up, A_down], [Omega_up, Omega_down]), U, spin)
        Sigma_HOMO_h = U**2 * (A_up/Omega_up + A_down/Omega_down) * dn**2
        Sigma_HOMO_e = U**2 * (A_up/(Omega_up + gap) + A_down/(Omega_down + gap)) * (dn**2 - 1)

        Sigma_HOMO = Sigma_HOMO_h + Sigma_HOMO_e
        Sigma_LUMO = -Sigma_HOMO

        HOMO = e[0] + Z * (Sigma_HOMO - vc_HOMO) ; HOMOs.append(HOMO)
        LUMO = e[1] + Z * (Sigma_LUMO - vc_LUMO) ; LUMOs.append(LUMO)

    return HOMOs[0], LUMOs[0]


def G0W0_QP(d, U, t, StartingPoint='KS', ExactKSHomo=True):
    if StartingPoint == 'KS':
        G0 = KS(d, U, t, ExactKSHomo=ExactKSHomo)
        return G0W0_correction(G0, d, U, t)
    elif StartingPoint == 'RHF':
        G0 = RHF(d, U, t)
        return G0W0_correction(G0, d, U, t)
    elif StartingPoint == 'UHF':
        G0 = UHF(d, U, t, dn_init=[-1, 1], tol=1e-6, maxiter=100, mixing=0.7, sort=False)
        return G0W0_correction_spin_unrestricted(G0, d, U, t)
