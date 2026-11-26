# Functions to compute the cooling to space

import numpy as np
from constants import g, cp, nu_rot, nu_vr, nu_co2, nu_min, nu_max, p0, cos_theta
import optical
import parameters



def cts_trans_nu(p_half, q_half, T, trans, nu):
    """
        Spectrally resolved CTS for a given transmissivity
    """  
    C = -np.pi * g / cp
      
    dtransp = (trans[1:] - trans[:-1]) / (p_half[1:].reshape(-1,1) - p_half[:-1].reshape(-1,1))
    HR_nu = C * optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * dtransp
    return HR_nu

def cts_trans(p_half, q_half, T, trans, nu):
    """
        Spectrally integrated CTS for a given transmissivity
    """
    delta_nu = nu[1:] - nu[:-1]
    integrand = cts_trans_nu(p_half, q_half, T, trans, nu)
    
    HR = np.sum((integrand[:,1:] + integrand[:,:-1])/2 * delta_nu, axis=-1)
    return HR

def cts_h2o(p_half, q_half, T):
    """
        Reference spectrally integrated CTS for H2O on each band.
        Sum the 4 bands to obtain cooling over the full spectrum.
    """
    pq = optical.local_pq(p_half, q_half)

    # Wavenumber grid
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))

    # Absorption bands
    nus_rot_left = nu[(nu <= nu_rot)]
    nus_rot_right = nu[(nu_rot <= nu) & (nu <= parameters.nu_mid)]
    nus_vr_left = nu[(parameters.nu_mid <= nu) & (nu <= nu_vr)]
    nus_vr_right = nu[(nu_vr <= nu)]
    
    C = -np.pi * g / cp
    # Grid spacing for the wavenumber grid
    delta_nu = nu[1] - nu[0]
    
    HR_band = np.zeros((4,pq.shape[0]))

    for k, nus in enumerate([nus_rot_left,nus_rot_right, nus_vr_left,nus_vr_right]):
        
        tau = np.insert(np.cumsum(pq.reshape(-1,1) * optical.k_h2o(nus).reshape(1,-1), axis = 0), 0, 0, axis=0)
        trans = np.exp(-tau)
        dtransp = (trans[1:] - trans[:-1]) / (p_half[1:].reshape(-1,1) - p_half[:-1].reshape(-1,1))
        integrand = C * optical.B(nus.reshape(1,-1), T.reshape(-1,1)) * dtransp
        
        # Cooling rate over each band
        HR_band[k] = np.sum((integrand[:,1:] + integrand[:,:-1])/2 * delta_nu, axis=-1)
    return HR_band

# tau = 1
def cts_tau1_h2o(p,p_half,q, q_half,T,T_half):
    """
        tau = 1 approximation of the spectrally integrated CTS for H2O on each band side.
        Sum the 4 band sides to obtain cooling over the full spectrum.
    """

    pq = optical.total_pq(p,q)
    nu = optical.tau_inverse_h2o(pq, 1)
    C = np.pi * p * q/(cp*p0*cos_theta*pq)

    mask0 = (nu[0] != -1)
    mask1 = (nu[1] != -1)
    mask2 = (nu[2] != -1)
    mask3 = (nu[3] != -1)

    H1 = np.zeros_like(p)
    H2 = np.zeros_like(p)
    H3 = np.zeros_like(p)
    H4 = np.zeros_like(p)

    H1[mask0] = C[mask0] * parameters.l_rot * optical.B(nu[0][mask0], T[mask0])
    H2[mask1] = C[mask1] * parameters.l_rot * optical.B(nu[1][mask1], T[mask1])
    H3[mask2] = C[mask2] * parameters.l_vr  * optical.B(nu[2][mask2], T[mask2])
    H4[mask3] = C[mask3] * parameters.l_vr  * optical.B(nu[3][mask3], T[mask3]) 
    
    HR_band = np.zeros((4,p.shape[0]))
    HR_band[0] = H1
    HR_band[1] = H2
    HR_band[2] = H3
    HR_band[3] = H4
    return HR_band

# midpoint
def cts_transmid_h2o(p,p_half,q, q_half,T,T_half):
    """
        Midpoint approximation of the spectrally integrated CTS for H2O on each band side.
        Sum the 4 band sides to obtain cooling over the full spectrum.
    """

    pq = optical.total_pq(p,q)
    C = np.pi * p * q/(cp*p0*cos_theta*pq)

    tau_rot_left_min = pq * optical.k_h2o(nu_min)
    tau_rot_right_min = pq * optical.k_h2o(parameters.nu_mid)
    tau_rot_max = pq * parameters.kappa_rot
    tau_vr_left_min = pq * optical.k_h2o(parameters.nu_mid)
    tau_vr_right_min = pq * optical.k_h2o(nu_max)
    tau_vr_max = pq * parameters.kappa_vr

    trans_nu = np.zeros((4,p.size))
    trans_nu[0] = (np.exp(-tau_rot_left_min) + np.exp(-tau_rot_max))/2
    trans_nu[1] = (np.exp(-tau_rot_right_min) + np.exp(-tau_rot_max))/2
    trans_nu[2] = (np.exp(-tau_vr_left_min) + np.exp(-tau_vr_max))/2
    trans_nu[3] = (np.exp(-tau_vr_right_min) + np.exp(-tau_vr_max))/2

    nu = optical.trans_inverse_h2o(pq, trans_nu)

    delta_rot_left = np.exp(-tau_rot_left_min) - np.exp(-tau_rot_max)
    delta_rot_right = np.exp(-tau_rot_right_min) - np.exp(-tau_rot_max)
    delta_vr_left = np.exp(-tau_vr_left_min) - np.exp(-tau_vr_max)
    delta_vr_right = np.exp(-tau_vr_right_min) - np.exp(-tau_vr_max)

    H1 = C * parameters.l_rot * optical.B(nu[0], T) * delta_rot_left
    H2 = C * parameters.l_rot * optical.B(nu[1], T) * delta_rot_right
    H3 = C * parameters.l_vr  * optical.B(nu[2], T) * delta_vr_left
    H4 = C * parameters.l_vr  * optical.B(nu[3], T) * delta_vr_right
    
    HR_band = np.zeros((4,p.shape[0]))
    HR_band[0] = H1
    HR_band[1] = H2
    HR_band[2] = H3
    HR_band[3] = H4
    return HR_band

def cts_co2(p_half, q_co2, T):
    """
        Reference spectrally integrated CTS for CO2 on each band.
        Sum the 2 band sides to obtain cooling over the full spectrum.
    """
    pq = optical.local_pq(p_half, q_co2 + np.zeros_like(p_half))

    # Wavenumber grid
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))

    # Absorption bands
    nus_co2_left = nu[(nu <= nu_co2)]
    nus_co2_right = nu[(nu_co2 <= nu)]
    
    C = -np.pi * g / cp
    # Grid spacing for the wavenumber grid
    delta_nu = nu[1] - nu[0]
    
    HR_band = np.zeros((2,pq.shape[0]))

    for k, nus in enumerate([nus_co2_left,nus_co2_right]):
        
        tau = np.insert(np.cumsum(pq.reshape(-1,1) * optical.k_co2(nus).reshape(1,-1), axis = 0), 0, 0, axis=0)
        trans = np.exp(-tau)
        dtransp = (trans[1:] - trans[:-1]) / (p_half[1:].reshape(-1,1) - p_half[:-1].reshape(-1,1))
        integrand = C * optical.B(nus.reshape(1,-1), T.reshape(-1,1)) * dtransp
        
        # Cooling rate over each band
        HR_band[k] = np.sum((integrand[:,1:] + integrand[:,:-1])/2 * delta_nu, axis=-1)
    return HR_band

# tau = 1
def cts_tau1_co2(p,p_half,q_co2,T,T_half):
    """
        tau = 1 approximation of the spectrally integrated CTS for CO2 on each band side.
        Sum the 2 band sides to obtain cooling over the full spectrum.
    """
    pq = optical.total_pq(p,q_co2 + np.zeros_like(p))
    nu = optical.tau_inverse_co2(pq, 1)
    C = np.pi * p * q_co2/(cp*p0*cos_theta*pq)

    mask0 = (nu[0] != -1)
    mask1 = (nu[1] != -1)
    
    H1 = np.zeros_like(p)
    H2 = np.zeros_like(p)

    H1[mask0] = C[mask0] * parameters.l_co2 * optical.B(nu[0][mask0], T[mask0])
    H2[mask1] = C[mask1] * parameters.l_co2 * optical.B(nu[1][mask1], T[mask1])
    
    HR_band = np.zeros((4,p.shape[0]))
    HR_band[0] = H1
    HR_band[1] = H2
    return HR_band

# midpoint
def cts_transmid_co2(p,p_half,q_co2,T,T_half):
    """
        Midpoint approximation of the spectrally integrated CTS for CO2 on each band side.
        Sum the 2 band sides to obtain cooling over the full spectrum.
    """
    pq = optical.total_pq(p,q_co2 + np.zeros_like(p))
    nu =  np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    C = np.pi * p * q_co2/(cp*p0*cos_theta*pq)

    mask1 = (nu < nu_co2)
    mask2 = (nu_co2 <= nu) 
    
    tau_co2_left_min = pq * optical.k_co2(nu_min)
    tau_co2_right_min = pq * optical.k_co2(nu_max)
    tau_co2_max = pq * parameters.kappa_co2

    delta_co2_left = (np.exp(-tau_co2_left_min) - np.exp(-tau_co2_max))
    delta_co2_right = (np.exp(-tau_co2_right_min) - np.exp(-tau_co2_max))

    trans_nu = np.zeros((2,p.size))
    trans_nu[0] = (np.exp(-tau_co2_left_min) + np.exp(-tau_co2_max))/2
    trans_nu[1] = (np.exp(-tau_co2_right_min) + np.exp(-tau_co2_max))/2

    nu = optical.trans_inverse_co2(pq, trans_nu)


    H1 = C * parameters.l_co2 * optical.B(nu[0], T) * delta_co2_left
    H2 = C * parameters.l_co2 * optical.B(nu[1], T) * delta_co2_right

    HR_band = np.zeros((2,p.shape[0]))
    HR_band[0] = H1
    HR_band[1] = H2
    return HR_band

