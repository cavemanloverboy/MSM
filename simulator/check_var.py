import numpy as np
import pandas as pd

def load_complex(file: str):
    # Load Data
    data = np.load(file)
    # Construct and return complex array
    return data["real"] + 1j*data["imag"]

def load_params(file: str):
    
    # Get name of hosting dir
    path_to_file = file.split("/")[:-1]

    # Append "/parameters.txt"
    #path_to_params = f"{'/'.join(path_to_file)}/parameters.txt"

    # return genfromtxt
    return []#pd.read_fwf(path_to_params, index_col=0,header=0)

def load_dump(file: str):

    return load_complex(file)#, load_params(file)

def compute_var(file: str):
    
    psi = load_dump(f"{file}/psi_00000")
    psi2 = load_dump(f"{file}/psi2_00000")
    mft = load_dump("sim-data/bose-star-512/psi_00000")

    ntot = 102090038309286.89
    dx = 0.1953125
    Ns = 64

    norm_check = np.sum(np.abs(psi)**2 * dx ** 3)

    mean_when_sq_then_mean = np.mean(psi2 - np.abs(mft)**2) * ntot * dx ** 3
    mean_when_mean_then_sq = np.mean(np.abs(psi - mft)**2) * ntot * dx ** 3
    var_when_sq_then_mean = np.var(psi2 - np.abs(mft)**2) * ntot * dx ** 3
    var_when_mean_then_sq = np.var(np.abs(psi - mft)**2) * ntot * dx ** 3

    print(f"{norm_check = }")
    print(f"{mean_when_sq_then_mean = }")
    print(f"{mean_when_mean_then_sq = }")
    print(f"{var_when_sq_then_mean = }")
    print(f"{var_when_mean_then_sq = }")
    print(f"{Ns * var_when_mean_then_sq = }")

    #print(((np.abs(psi)**2 - np.abs(mft)**2) * dx**3).sum())


if __name__ == "__main__":
    file = "sim-data/bose-star-512-combined"
    compute_var(file)
