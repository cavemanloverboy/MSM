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
    path_to_params = f"{'/'.join(path_to_file)}/parameters.txt"

    # return genfromtxt
    return pd.read_fwf(path_to_params, index_col=0,header=0)

def load_dump(file: str):

    return load_complex(file), load_params(file)

def compute_var(file: str):
    
    psi, params = load_dump(file)
    mft, _ = load_dump("sim_data/bose-star-512/psi_00000")
    print(params.keys())
    ntot = 102090038309286.89#params["ntot"]
    dx = 0.1953125
    psi2 = np.abs(psi-mft)**2 * ntot * dx**3

    print((np.abs(psi)**2 * dx**3).sum())
    print(psi2.sum() / ntot)
    print((psi * np.sqrt(ntot) * dx**(3/2)).std())
    print(psi2.std())


if __name__ == "__main__":
    file = "sim_data/bose-star-512-stream00000/psi_00000"
    compute_var(file)
