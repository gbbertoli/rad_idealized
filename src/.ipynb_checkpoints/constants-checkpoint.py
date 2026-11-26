import scipy

h_pl = 6.62607015e-34 # Planck constant [J s]
c_light = 299792458 # Speed of light [m/s]
kb = 1.380649e-23 # Boltzmann constant [J/K]
Rd = 287 # Specific gas constant of dry air [J/(Kg K)]
Rv = 461.5 # Specific gas constant of water vapor [J/(Kg K)]
T0 = 300 # Arbitrary reference temperature [K] 
e0 = 3525 # Saturation vapor pressure at T0 [Pa]
gamma_wv = 18 # Power law exponent of temperature - satuation vapor pressure relation []
Lv = 2492100 # Latent heat of vaporization of water at T0 [J/kg]
ps = 101325 # Mean sea level pressure [Pa]
cp = 1003.5 # Specific heat of dry air at constant pressure [J / (Kg K)]
g = 9.81 # Average gravity on Earth [m/s^2]
pi = scipy.constants.pi

nu_rot = 150 * 100 # Middle of the rotation band [1/m]
nu_vr = 1500 * 100 # Middle of the vibration-rotation band [1/m]
nu_co2 = 667.5 * 100 # Middle of the CO2 band [1/m]

cos_theta = 3/5 # Inverse angular diffusivity factor []
p0 = 100000 # Reference pressure for the absorption coefficients [Pa]
nu_min = 1 * 100
nu_max = 2000 * 100