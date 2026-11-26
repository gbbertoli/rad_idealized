# H2O and CO2 idealized spectroscopy

import numpy as np
from constants import g, p0, h_pl, c_light, kb, nu_rot, nu_vr, nu_co2, cos_theta, nu_min, nu_max
import parameters

# kappa_rot = parameters.kappa_rot
# l_rot = parameters.l_rot
# kappa_vr = parameters.kappa_vr
# l_vr = parameters.l_vr
# kappa_co2 = parameters.kappa_co2
# l_co2 = parameters.l_co2

def k_h2o(nu):
    return np.max((parameters.kappa_rot * np.exp(-np.abs(nu-nu_rot)/parameters.l_rot), parameters.kappa_vr * np.exp(-np.abs(nu-nu_vr)/parameters.l_vr)),axis=0)

def k_co2(nu):
    return parameters.kappa_co2 * np.exp(-np.abs(nu-nu_co2)/parameters.l_co2)

# Planck function as function of wavenumber [1/cm] and temperature [K]
def B(nu, T):
    return 2 * h_pl * c_light ** 2 * nu ** 3 /(np.exp(h_pl * c_light * nu / (kb * T)) - 1)

# Total atmospheric path
def total_pq(p, q):
    delta_p = np.concatenate((p[0].reshape(1), p[1:] - p[:-1]))
    trapeizodal_pq = np.concatenate(((p[0]*q[0]/2).reshape(1), (p[:-1]*q[:-1] + p[1:]*q[1:])/2))
    return np.cumsum((1/(g * p0 * cos_theta) * delta_p * trapeizodal_pq))

# Atmospheric path for a single layer
def local_pq(p, q):
    delta_p =  p[1:] - p[:-1]
    trapeizodal_pq = (p[:-1]*q[:-1] + p[1:]*q[1:])/2
    return 1/(g * p0 * cos_theta) * delta_p * trapeizodal_pq
    
# All tau corresponding to 
def tau_inverse_h2o(pq, tau):
    nu1 = - parameters.l_rot * np.log(pq * parameters.kappa_rot) + nu_rot + parameters.l_rot * np.log(tau) 
    nu2 =  parameters.l_rot * np.log(pq * parameters.kappa_rot) + nu_rot - parameters.l_rot * np.log(tau)
    nu3 = - parameters.l_vr * np.log(pq * parameters.kappa_vr) + nu_vr + parameters.l_vr * np.log(tau)
    nu4 =  parameters.l_vr * np.log(pq * parameters.kappa_vr) + nu_vr - parameters.l_vr * np.log(tau)
    mask1 = (nu_min < nu1) & (nu1 < nu_rot)
    mask2 = (nu_rot <= nu2) & (nu2 < parameters.nu_mid)
    mask3 = (parameters.nu_mid <= nu3) & (nu3 < nu_vr)
    mask4 = (nu_vr <= nu4) & (nu4 < nu_max)
    nu1[~mask1] = -1;
    nu2[~mask2] = -1;
    nu3[~mask3] = -1;
    nu4[~mask4] = -1;
    return np.stack([nu1, nu2, nu3, nu4],axis=0)

def tau_inverse_co2(pq, tau):
    nu1 = - parameters.l_co2 * np.log(pq * parameters.kappa_co2) + nu_co2 + parameters.l_co2 * np.log(tau) 
    nu2 =  parameters.l_co2 * np.log(pq * parameters.kappa_co2) + nu_co2 - parameters.l_co2 * np.log(tau)
    mask1 = (nu_min < nu1) & (nu1 < nu_co2)
    mask2 = (nu_co2 <= nu2) & (nu2 < nu_max)
    nu1[~mask1] = -1;
    nu2[~mask2] = -1;
    return np.stack([nu1, nu2],axis=0)

def trans_inverse_h2o(pq, trans):
    nu1 = - parameters.l_rot * np.log(pq * parameters.kappa_rot) + nu_rot + parameters.l_rot * np.log(np.abs(np.log(trans[0])))
    nu2 =  parameters.l_rot * np.log(pq * parameters.kappa_rot) + nu_rot - parameters.l_rot * np.log(np.abs(np.log(trans[1])))
    nu3 = - parameters.l_vr * np.log(pq * parameters.kappa_vr) + nu_vr + parameters.l_vr * np.log(np.abs(np.log(trans[2])))
    nu4 =  parameters.l_vr * np.log(pq * parameters.kappa_vr) + nu_vr - parameters.l_vr * np.log(np.abs(np.log(trans[3])))
    mask1 = (nu_min < nu1) & (nu1 < nu_rot)
    mask2 = (nu_rot <= nu2) & (nu2 < parameters.nu_mid)
    mask3 = (parameters.nu_mid <= nu3) & (nu3 < nu_vr)
    mask4 = (nu_vr <= nu4) & (nu4 < nu_max)
    nu1[~mask1] = -1;
    nu2[~mask2] = -1;
    nu3[~mask3] = -1;
    nu4[~mask4] = -1;
    return np.stack([nu1, nu2, nu3, nu4],axis=0)

def trans_inverse_co2(pq, trans):
    nu1 = - parameters.l_co2 * np.log(pq * parameters.kappa_co2) + nu_co2 + parameters.l_co2 * np.log(np.abs(np.log(trans[0])))
    nu2 =  parameters.l_co2 * np.log(pq * parameters.kappa_co2) + nu_co2 - parameters.l_co2 * np.log(np.abs(np.log(trans[1])))
    mask1 = (nu1 < nu_co2)
    mask2 = (nu_co2 <= nu2)
    nu1[~mask1] = -1;
    nu2[~mask2] = -1;
    return np.stack([nu1, nu2],axis=0)
    
def trans_malkmus(tau, gamma):
        if isinstance(gamma, np.ndarray):
            gamma = gamma.reshape(-1,1)
        c = np.pi * gamma / 4
        tau_hat = np.sqrt((tau + c) * c)
        return np.exp(-2 * (tau_hat - c))