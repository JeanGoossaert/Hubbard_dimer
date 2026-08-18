import numpy as np
import matplotlib.pyplot as plt



### RHF ###

def RHF_potential(d, U, dn):
    n1 = (2+dn)/2 ; n2 = (2-dn)/2
    return [-d/2 + U*n1/2, d/2 + U*n2/2]

def RHF_dn(dveff, t):
    return 2*dveff / np.sqrt(4*t**2  + dveff**2)

def RHF_1iteration(d, U, dn, t):
    veff = RHF_potential(d, U, dn)
    dveff = veff[1] - veff[0]
    return RHF_dn(dveff, t)

def RHF_selfconsistent(d, U, dn_init, t, tol=1e-6, maxiter=100, mixing=0.3):
    dn = dn_init
    for i in range(maxiter):
        dn_new = RHF_1iteration(d, U, dn, t)
        dn_new = mixing * dn_new + (1 - mixing) * dn
        if abs(dn_new - dn) < tol:
            #print(f"RHF converged in {i+1} iterations.")
            return dn_new
        dn = dn_new
    print("RHF did not converge within the maximum number of iterations.")
    return dn

def RHF_energies(d, U, dn, t):
    veff = RHF_potential(d, U, dn)
    dveff = veff[1] - veff[0] ; veff_avg = (veff[0] + veff[1])/2
    e0_RHF = -0.5* np.sqrt((2*t)**2 + dveff**2) + veff_avg
    e1_RHF = 0.5* np.sqrt((2*t)**2 + dveff**2) + veff_avg
    return [e0_RHF, e1_RHF], veff

def RHF(d, U, t, dn_init=0,tol=1e-6, maxiter=100, mixing=0.3):
    dn = RHF_selfconsistent(d, U, dn_init, t, tol, maxiter, mixing)
    energies, veff = RHF_energies(d, U, dn, t)
    return energies, veff, dn


### UHF ###

def UHF_potential(d, U, dn_sigma):
    [dn_up, dn_down] = dn_sigma
    n1_up = (1+dn_up)/2 ; n2_up = (1-dn_up)/2
    n1_down = (1+dn_down)/2 ; n2_down = (1-dn_down)/2
    veff_up = [-d/2 + U*n1_down, d/2 + U*n2_down]
    veff_down = [-d/2 + U*n1_up, d/2 + U*n2_up]
    return veff_up, veff_down

def UHF_1iteration(d, U, dn_sigma, t):
    veff_up, veff_down = UHF_potential(d, U, dn_sigma)
    dveff_up = veff_up[1] - veff_up[0] ; dveff_down = veff_down[1] - veff_down[0]
    return RHF_dn(dveff_up, t)/2, RHF_dn(dveff_down, t)/2

def UHF_selfconsistent(d, U, dn_init, t, tol=1e-6, maxiter=100, mixing=0.5):
    dn_sigma = dn_init
    for i in range(maxiter):
        dn_up_new, dn_down_new = UHF_1iteration(d, U, dn_sigma, t)
        dn_up_new = mixing * dn_up_new + (1 - mixing) * dn_sigma[0] ; dn_down_new = mixing * dn_down_new + (1 - mixing) * dn_sigma[1]
        if abs(dn_up_new - dn_sigma[0]) < tol and abs(dn_down_new - dn_sigma[1]) < tol:
            #print(f"UHF converged in {i+1} iterations.")
            return [dn_up_new, dn_down_new]
        dn_sigma = [dn_up_new, dn_down_new]
    print("UHF did not converge within the maximum number of iterations.")
    return dn_sigma

def UHF_energies(d, U, dn_sigma, t):
    [dn_up, dn_down] = dn_sigma
    [e0_up, e1_up], veff_up = RHF_energies(d, U, dn_down*2, t) # veff_up is constructed with dn_down, and vice-versa
    [e0_down, e1_down], veff_down = RHF_energies(d, U, dn_up*2, t)
    return [e0_up, e1_up, e0_down, e1_down], [veff_up, veff_down]

def UHF(d, U, t, dn_init=[1, -1], tol=1e-6, maxiter=100, mixing=0.7, sort=False):
    dn_sigma = UHF_selfconsistent(d, U, dn_init, t, tol, maxiter, mixing)
    energies, veff = UHF_energies(d, U, dn_sigma, t)
    
    if sort:
        energies_sorted_indices = np.argsort(energies)
        energies = [energies[i] for i in energies_sorted_indices]
        veff = [veff[i//2] for i in energies_sorted_indices]
        dn_sigma = [dn_sigma[i//2] for i in energies_sorted_indices]
        return energies[1:3], veff[1:3], dn_sigma[1:3] # only return HOMO and LUMO
    else:
        return energies, veff, dn_sigma