# Functions to compute the outgoing longwave radiation

import numpy as np
import optical
import parameters
from constants import nu_min, nu_max


def olr_trans(p_half, q_half, T, T_half, trans, nu):
    """
        OLR for a given transmissivity
    """
    
    integrand = np.pi * optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans[:-1] - trans[1:])
    olr_atm = np.sum(integrand, axis=0)
    olr_surf = (np.pi * optical.B(nu.reshape(1,-1), T_half[-1].reshape(-1,1)) * trans[-1].reshape(1,-1)).ravel()
    return olr_atm, olr_surf

def olr_h2o(p_half, q_half, T, Ts):
    """
        Reference OLR for H2O
    """
    pq = optical.local_pq(p_half, q_half)
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = pq.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)
    tau = np.cumsum(tau_local, axis = 0)
    tau = np.insert(tau, 0, 0, axis = 0)
    trans = np.exp(-tau)
    integrand = np.pi * optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans[:-1] - trans[1:]) 
    
    olr_atm = np.sum(integrand, axis=0)
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]
    return olr_atm, olr_surf


def olr_tau1_h2o(p_half, q_half, T, Ts):
    """
        tau = 1 approximation of the OLR for H2O
    """
    pq = optical.local_pq(p_half, q_half)
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = pq.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)
    tau = np.cumsum(tau_local, axis = 0)

    mask = tau[-1] >= 1
    
    i_tau1 = np.argmin(np.abs(tau - 1), axis = 0)
    p_nu = np.stack([i_tau1[mask], np.arange(0,tau.shape[1])[mask]])
    
    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) 
    integrand_surf = np.pi * optical.B(nu, Ts)

    olr_atm = np.zeros(integrand.shape[1])
    olr_surf = np.zeros(integrand.shape[1])

    olr_atm[mask] = integrand[*p_nu]
    olr_surf[~mask] = integrand_surf[~mask]
    return olr_atm, olr_surf

def olr_transmid_h2o(p_half, q_half, T, Ts):
    """
        Midpoint approximation of the OLR for H2O
    """
    pq = optical.local_pq(p_half, q_half)
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = pq.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)
    tau = np.cumsum(tau_local, axis = 0)
    
    trans = np.exp(-tau)
    trans_max = trans[0,:]
    trans_min = trans[-1,:]

    i_mid = np.argmin(np.abs(trans - (trans_max + trans_min)/2), axis = 0)
    p_nu = np.stack([i_mid, np.arange(0,tau.shape[1])])
    
    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans_max - trans_min)
    olr_atm = integrand[*p_nu]
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]

    return olr_atm, olr_surf

def olr_co2(p_half, qco2, T, Ts):
    """
        Reference OLR for CO2
    """
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(p_half))

    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = (pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1) )
    tau = np.cumsum(tau_local, axis = 0)
    tau = np.insert(tau, 0, 0, axis = 0)
    trans = np.exp(-tau)
    integrand = np.pi * optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans[:-1] - trans[1:]) 
    olr_atm = np.sum(integrand, axis=0)
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]

    return olr_atm, olr_surf
    
def olr_tau1_co2(p_half, qco2, T, Ts):
    """
        tau = 1 approximation of the OLR for CO2
    """
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(p_half))
    
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1) 
    tau = np.cumsum(tau_local, axis = 0)

    mask = tau[-1] >= 1
    
    i_tau1 = np.argmin(np.abs(tau - 1), axis = 0)
    p_nu = np.stack([i_tau1[mask], np.arange(0,tau.shape[1])[mask]])

    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) 
    integrand_surf = np.pi * optical.B(nu, Ts)

    olr_atm = np.zeros(integrand.shape[1])
    olr_surf = np.zeros(integrand.shape[1])

    olr_atm[mask] = integrand[*p_nu]
    olr_surf[~mask] = integrand_surf[~mask]
    return olr_atm, olr_surf

def olr_transmid_co2(p_half, qco2, T, Ts):
    """
        Midpoint approximation of the OLR for CO2
    """
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(p_half))
    
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1) 
    tau = np.cumsum(tau_local, axis = 0)
    
    trans = np.exp(-tau)
    trans_max = trans[0,:]
    trans_min = trans[-1,:]

    i_mid = np.argmin(np.abs(trans - (trans_max + trans_min)/2), axis = 0)
    p_nu = np.stack([i_mid, np.arange(0,tau.shape[1])])
    
    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans_max - trans_min)

    olr_atm = integrand[*p_nu]
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]
    return olr_atm, olr_surf

    
def olr_h2o_co2(p_half, q_half, qco2, T, Ts):
    """
        Reference OLR for an atmosphere with H2O and CO2
    """
    pq_h2o_band = optical.local_pq(p_half, q_half)
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(q_half))

    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = (pq_h2o_band.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)  
               + pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1))
    tau = np.cumsum(tau_local, axis = 0)
    tau = np.insert(tau, 0, 0, axis = 0)
    trans = np.exp(-tau)
    integrand = np.pi * optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans[:-1] - trans[1:])
    olr_atm = np.sum(integrand, axis=0)
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]
    return olr_atm, olr_surf

def olr_tau1_h2o_co2(p_half, q_half, qco2, T, Ts):
    """
        tau = 1 approximation of the OLR for an atmosphere with H2O and CO2
    """
    pq_h2o_band = optical.local_pq(p_half, q_half)
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(q_half))
    
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = (pq_h2o_band.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)  
               + pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1))
    tau = np.cumsum(tau_local, axis = 0)

    mask = tau[-1] >= 1
    i_tau1 = np.argmin(np.abs(tau - 1), axis = 0)
    p_nu = np.stack([i_tau1[mask], np.arange(0,tau.shape[1])[mask]])
    
    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) 
    integrand_surf = np.pi * optical.B(nu, Ts)

    olr_atm = np.zeros(integrand.shape[1])
    olr_surf = np.zeros(integrand.shape[1])

    olr_atm[mask] = integrand[*p_nu]
    olr_surf[~mask] = integrand_surf[~mask]
    return olr_atm, olr_surf

def olr_transmid_h2o_co2(p_half, q_half, qco2, T, Ts):
    """
        Midpoint approximation of the OLR for an atmosphere with H2O and CO2
    """    
    pq_h2o_band = optical.local_pq(p_half, q_half)
    pq_co2 = optical.local_pq(p_half, qco2 + np.zeros_like(q_half))
    
    nu = np.linspace(nu_min, nu_max, int((nu_max - nu_min + 1)/100))
    tau_local = (pq_h2o_band.reshape(-1,1) * optical.k_h2o(nu).reshape(1,-1)  
               + pq_co2.reshape(-1,1) * optical.k_co2(nu).reshape(1,-1))
    tau = np.cumsum(tau_local, axis = 0)
    
    trans = np.exp(-tau)
    trans_max = trans[0,:]
    trans_min = trans[-1,:]

    i_mid = np.argmin(np.abs(trans - (trans_max + trans_min)/2), axis = 0)
    p_nu = np.stack([i_mid, np.arange(0,tau.shape[1])])
    
    integrand =  np.pi* optical.B(nu.reshape(1,-1), T.reshape(-1,1)) * (trans_max - trans_min)

    olr_atm = integrand[*p_nu]
    olr_surf = np.pi * optical.B(nu, Ts) * trans[-1]
    return olr_atm, olr_surf
