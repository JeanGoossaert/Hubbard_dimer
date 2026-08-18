import numpy as np



def ExpectationValue(G0, state, Operator, t):
    e, veff_list, dn = G0
    [[M11, M12], [M21, M22]] = Operator

    dveff = veff_list[1] - veff_list[0]
    rho = np.arctan( -dveff/(2*t) + np.sqrt(1+(dveff/(2*t))**2) )
    s=np.sin(rho) ; c = np.cos(rho)

    if state == 0:
        a = c**2 * M11 + s*c*(M12 + M21) + s**2 * M22
    elif state == 1:
        a = s**2 * M11 - s*c*(M12 + M21) + c**2 * M22
    return a


def Correlation_potential(G0, d, U, SpinUnrestricted=False, dn_bar=None):
    e, veff_list, dn = G0
    veff1, veff2 = veff_list
    if SpinUnrestricted:
        n1 = (1 + dn_bar) / 2 ; n2 = (1 - dn_bar) / 2 # the Hartree potential uses dn_down for the spin up (and vice-versa)
        return veff1 - (-d/2 + U * n1), veff2 - (d/2 + U * n2)
    else:
        n1 = (2 + dn) / 2 ; n2 = (2 - dn) / 2
        return veff1 - (-d/2 + U * n1/2), veff2 - (d/2 + U * n2/2)


## 1-electron spin-degenerate (KS, RHF) eigenstates in the site basis
def Single_particle_restricted_eigenstates(veff_list, t):
    dveff = veff_list[1] - veff_list[0]
    rho = np.arctan( -dveff/(2*t) + np.sqrt(1+(dveff/(2*t))**2) )
    s=np.sin(rho) ; c = np.cos(rho)

    return np.array([[c, s], [s, -c]])

## 1-electron spin-unrestricted (UHF) eigenstates in the site basis
def Single_particle_unrestricted_eigenstates(veff_list_sigma, t):
    EigenStates = []
    for spin,veff_list in enumerate(veff_list_sigma):
        dveff = veff_list[1] - veff_list[0]
        rho = np.arctan( -dveff/(2*t) + np.sqrt(1+(dveff/(2*t))**2) )
        s=np.sin(rho) ; c = np.cos(rho)
        EigenStates.append([c, s]) ; EigenStates.append([s, -c])

    return np.array(EigenStates)


def SortEigenstates(enerlist, eigenvect):
    idx = np.argsort(enerlist)
    enerlist_ = np.array(enerlist)[idx]
    eigenvect_ = np.array(eigenvect)[:,idx]
    return enerlist_, eigenvect_