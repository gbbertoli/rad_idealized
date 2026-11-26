from contextlib import contextmanager
import numpy as np
import pandas as pd


def parse_atmosphere_file(filepath):
    """
    Parse an atmospheric profile file.
    Returns a DataFrame with the profile data and a dict of metadata.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Parse the header line ---
    header = lines[0].split()
    n_levels      = int(header[0])
    T_surface     = float(header[5])
    P_surface     = float(header[6])

    metadata = {
        "n_levels":    n_levels,
        "T_surface":   T_surface,
        "P_surface":   P_surface,
    }

    # --- Parse the data lines ---
    records = []
    for line in lines[1:n_levels + 1]:
        values = line.split()
        records.append({
            "pressure":          float(values[0]),
            "temperature":       float(values[1]),
            "specific_humidity": float(values[2]),
            "o3_mass_fraction":  float(values[3]),
        })

    df = pd.DataFrame(records)
    return df, metadata


def atmosphere_4a():
    """
        Read the line-by-line file to recover the pressure, temperature and specific humidity
        at the layer interfaces.
        Compute the pressure, temperature and specific humidity at the layer center with the mean.
    """
    
    # df_atm, meta = parse_atmosphere_file("lbl_data/atm4afmt_Koll2023.dsf")
    df_atm, meta = parse_atmosphere_file("../lbl_data/atm4afmt_idealized_atmosphere.dsf")
    p_half          = df_atm["pressure"].values * 100        # shape: (n_levels,)
    T_half       = df_atm["temperature"].values.copy()     # shape: (n_levels,)
    q_half = df_atm["specific_humidity"].values # shape: (n_levels,)
    p_half[-1] = meta["P_surface"] * 100
    T_half[-1] = meta["T_surface"]
    p = (p_half[1:] + p_half[:-1])/2
    T = (T_half[1:] + T_half[:-1])/2
    q = (q_half[1:] + q_half[:-1])/2
    return T, T_half, q, q_half, p, p_half


@contextmanager
def open_txt(chemin, mode='r', encoding='utf-8'):
    if chemin.endswith('.gz'):
        with gzip.open(chemin, mode + 't', encoding=encoding) as f:
            yield f
    else:
        with open(chemin, mode, encoding=encoding) as f:
            yield f

def parse_fortran_float(val_str):
    """
    Converts a Fortran string to a Python float.
    Handles ‘NaN’ values and scientific notation that omits the ‘E’.
    """
    val_str = val_str.strip()
    if val_str.upper() == 'NAN':
        return float('nan')
    
    if 'E' not in val_str.upper():
        val_str = re.sub(r'(?<=\d)([-+])(?=\d+)', r'E\1', val_str)
        
    return float(val_str)

def parse_fortran_float_line(line):
    """
    Converts a Fortran string to a Python float.
    Handles ‘NaN’ values and scientific notation that omits the ‘E’.
    """
    parts = line.split()
    x_tmp = np.zeros(len(parts), dtype="float64")
    for ip, part in enumerate(parts):
        x_tmp[ip] = parse_fortran_float(part)

    return x_tmp

def read_properties(fname):
    """read the dto 4A file, and provides the numbert of lines,
    vertical levels, and frequences in the file
    """
    with open_txt(fname) as df:
        line = df.readline().strip()
        tot_nb_lines = 0
        tot_nb_freqs = 0
        nb_blocks = 0

        while line:
            parts = line.split()
            tot_nb_lines += 1
            if parts[0] != "#":
                print("ERROR, line should start with #, line:",tot_nb_lines)
                print(line)
                sys.exit()
            nb_freqs = int(parts[1])
            nb_levs = int(parts[2])
            nb_blocks += 1

            # read the top, mid, bottom pressure of each level
            # read the transmissions
            nb_pres_ar = 3
            for jp in range(nb_pres_ar+nb_freqs):
                line = df.readline().strip()
            tot_nb_freqs += nb_freqs
            tot_nb_lines += nb_pres_ar+nb_freqs
            line = df.readline().strip()

    prop = {"nb_lines":tot_nb_lines, "nb_levs": nb_levs
            ,"nb_freqs":tot_nb_freqs, "nb_blocks": nb_blocks}
    return prop

def read_tau(fname):

    prop = read_properties(fname)
    nb_pres_ar = 3
    pres = np.full([prop["nb_levs"],nb_pres_ar],float("nan"))
    tau = np.full([prop["nb_levs"]-1,prop["nb_freqs"]],float("nan"),dtype="float64")
    trans = np.full([prop["nb_levs"]-1,prop["nb_freqs"]],float("nan"))
    nu = np.full(prop["nb_freqs"],float("nan"))
                    
    jf_cum = 0 # index of the concatenated frequencies
    
    with open_txt(fname) as df:

        for jblock in range(prop["nb_blocks"]):

            # read line starting with #
            line = df.readline().strip()
            parts = line.split()
            nb_freqs = int(parts[1])

            # read the top, mid, bottom pressure of each level
            for jp in range(nb_pres_ar):
                line = df.readline().strip()
                if jblock == 0:
                    pres[:,jp] = parse_fortran_float_line(line)
                
            # read the transmissions, compute the optical thickness
            for jf in range(nb_freqs):
                line = df.readline().strip()
                xxx =  parse_fortran_float_line(line)
                nu[jf_cum] = xxx[0]
                # tau[:,jf_cum] = -np.log(np.maximum(xxx[2:],0))
                # trans[:,jf_cum] = xxx[2:]
                tau[:,jf_cum] = xxx[2:]*1.E6
                trans[:,jf_cum] = np.exp(-tau[:,jf_cum])
                jf_cum += 1
    return nu,tau, trans, pres
