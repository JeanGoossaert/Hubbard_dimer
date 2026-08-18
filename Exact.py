import numpy as np


def Non_interacting_density(dvs, t):
    return 2*dvs / np.sqrt(4*t**2  + dvs**2)


## Returns the exact energies and coefficients for the 2-site Hubbard model for N=1,2,3 electrons
def Exact_energies(d, U, t):
    
    d/=t ; U/=t
    z_2 = 9*d**2 - U**2 - 18
    r = np.sqrt(3*d**2 + U**2 + 12)
    theta=(1/3)*np.arccos(z_2*U/(r**3))
    e0 = (2/3)*(U - r*np.cos(theta-np.pi/3)) ; e0*=t
    e1 = 0
    e2 = (2/3)*(U - r*np.cos(theta+np.pi/3)) ; e2*=t
    e3 = (2/3)*(U - r*np.cos(theta+np.pi)) ; e3*=t
    d*=t ; U*=t

    e0_N1 = -0.5*np.sqrt(4*t**2 + d**2) ; e1_N1 = -e0_N1
    e0_N3 = e0_N1 + U ; e1_N3 = e1_N1 + U


    a0 = 2*t/(e0-d-U) ; a2 = 2*t/(e2-d-U) ; a3 = 2*t/(e3-d-U)
    Norm0 = np.sqrt( 2*(1 + 2*(a0/2-e0/(2*t))**2 + 2*(a0/2)**2 ) )
    Norm2 = np.sqrt( 2*(1 + 2*(a2/2-e2/(2*t))**2 + 2*(a2/2)**2 ) )
    Norm3 = np.sqrt( 2*(1 + 2*(a3/2-e3/(2*t))**2 + 2*(a3/2)**2 ) )

    return [e0,e1,e2,e3], [e0_N1, e1_N1], [e0_N3, e1_N3], [a0,Norm0], [a2,Norm2], [a3,Norm3]



def Exact_DeltaN(d, U, t):
    d /= t ; U /= t
    z_2 = 9*d**2 - U**2 - 18
    r = np.sqrt(3*d**2 + U**2 + 12)
    theta=(1/3)*np.arccos(z_2*U/(r**3))
    e1 = (2/3)*(U - r*np.cos(theta-np.pi/3)) ; e1*=t
    U *= t ; d *= t

    uu = U/(2*t) ; nu = d/(2*t) ; eps = e1 /(2*t)
    c2 = 1/(2 * (nu**2 + (eps-uu)**2 *(1+(1/eps**2))) )

    R = 0.5* np.sqrt((2*t)**2 + d**2)
    return 4 * c2 * nu * (uu - eps) *2



def ExactChi11(d, U, t):
    [e0,e1,e2,e3], [e0_N1, e1_N1], [e0_N3, e1_N3], [a0,Norm0], [a2,Norm2], [a3,Norm3] = Exact_energies(d, U, t)

    ## Amplitudes
    # n = 1 does not contribute
    A2 = 4 * ((1 + (a0-e0/t)*(a2-e2/t)) / (Norm0 * Norm2))**2
    A3 = 4 * ((1 + (a0-e0/t)*(a3-e3/t)) / (Norm0 * Norm3))**2

    ## Poles
    Omega_2 = e2 - e0 ; Omega_3 = e3 - e0

    return [A2, A3], [Omega_2, Omega_3]



