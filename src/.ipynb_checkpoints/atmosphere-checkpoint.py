# Functions to define an idealized atmosphere based on
# Koll, Jeevanjee, Lutsko (2023), An Analytic Model for the Clear-Sky Longwave Feedback, 
# Journal of Atmospheric Sciences. DOI: 10.1175/JAS-D-22-0178.1

import numpy as np
from constants import Rd, Rv, e0, ps, Lv, gamma_wv, cp, T0, g

def Specific_humidity(p, Ts, Ttp, RH):
    """
    Specific humidity of an idealized atmosphere as function of surface temperature,
    constant relative humidity and tropopause temperature following Koll et al. (2023).

    Parameters
    ----------
    p : array
        Pressure

    Ts : float
        Temperature of the surface

    Ttp : float
        Temperature of the tropopause

    RH : float
        Relative humidity

    Returns
    -------
    q : array
        Specific humidity at pressures p
    """
    
    Tav = (Ts + Ttp)/2
    
    # Saturation specific humidity
    qs_sat = Rd/Rv * e0/ps * (Ts/T0)**gamma_wv
    
    # Dry lapse rate
    gamma_lr = Rd * Tav * np.log(Ts/Ttp) / (cp * (Ts - Ttp) + Lv * qs_sat)
    
    T = Ts * (p/ps) ** gamma_lr 
    q = RH * Rd/Rv * e0/p * (T/T0)**gamma_wv
    return q

def Temperature(p, Ts, Ttp):
    """
    Temperature of an idealized atmosphere as function of surface temperature 
    and tropopause temperature following Koll et al. (2023).
    The stratosphere is assumed isothermal and equal to the tropopause temperature.

    Parameters
    ----------
    p : array
        Pressure

    Ts : float
        Temperature of the surface

    Ttp : float
        Temperature of the tropopause

    Returns
    -------
    T : array
        Temperature at pressures p
    """
    Tav = (Ts + Ttp)/2
    T = np.ones_like(p) * Ttp
    
    # Saturation specific humidity
    qs_sat = Rd/Rv * e0/ps * (Ts/T0)**gamma_wv
    
    # Dry lapse rate
    gamma_lr = Rd * Tav * np.log(Ts/Ttp) / (cp * (Ts - Ttp) + Lv * qs_sat)

    # Scale height of pressure
    H = Rd * Tav / g
    
    # Tropopause height [m]
    z_tp = (cp * (Ts - Ttp) + Lv * qs_sat)/g 
    
    # Pressure of the tropopause [Pa]
    p_tp = ps* np.exp(-z_tp/H) 
    
    T[p>p_tp] = Ts * (p[p>p_tp]/ps) ** gamma_lr
    return T

def atmosphere(Ts, RH):
    """
    Pressure, Temperature and specific humidty of an idealized atmosphere as
    function of surface temperature and constant relative humidity
    following Koll et al. (2023)

    Parameters
    ----------
    RH : float
        Relative humidity

    Ts : float
        Temperature of the surface

    Returns
    -------
    T : array
        Temperature on full pressure levels
        
    T_half : array
        Temperature on half pressure levels

    q : array
        Specific humidity on full pressure levels
        
    q_half : array
        Specific humidity on half pressure levels

    p : array
        Pressure at full levels
        
    p_half : array
        Pressure at half levels
    """
    # Number of atmopheric layers = number of full pressure levels
    number_intervals = 100

    # Tropopause temperature
    Ttp = 2/3 * Ts

    # Pressure at the top of the atmosphere
    p_top = 1 
    
    p_half = np.linspace(p_top,ps, number_intervals + 1)
    p = (p_half[1:] + p_half[:-1])/2
    
    T_half = Temperature(p_half, Ts, Ttp)
    q_half = Specific_humidity(p_half, Ts, Ttp, RH)
    
    T = (T_half[1:] + T_half[:-1])/2
    q = (q_half[1:] + q_half[:-1])/2
    return T, T_half, q, q_half, p, p_half

def tropopause(Ts):
    """
    Pressure of the tropopause

    Parameters
    ----------
    ps : float
        pressure of the surface

    Ts : float
        temperature of the surface

    Returns
    -------
    p_tp : float
        pressure of the tropopause
    """
    Ttp = 2/3 * Ts    
    Tav = (Ts + Ttp)/2
    
    # Saturation specific humidity
    qs_sat = Rd/Rv * e0/ps * (Ts/T0)**gamma_wv
    
    # Dry lapse rate
    gamma_lr = Rd * Tav * np.log(Ts/Ttp) / (cp * (Ts - Ttp) + Lv * qs_sat)

    # Scale height of pressure
    H = Rd * Tav / g
    
    # Tropopause height [m]
    z_tp = (cp * (Ts - Ttp) + Lv * qs_sat)/g 
    
    # Pressure of the tropopause [Pa]
    p_tp = ps* np.exp(-z_tp/H) 
    return p_tp