import numpy as np 
from scipy import interpolate as s
import os
import scipy.fftpack as sp
import matplotlib.pyplot as plt 
# import husimi_sp as hu
# import di_analyze2D

# all units in kpc, Msolar, Myr
sim_name = "planeWave3d_e10_sym"
toml_dir = "tomls"
ics_dir = "initial_conditions"
N = 16#256
drops = 64 
a_ini = .01
T = 2000. # Myr
L = 60. # start or end length? 
dx = L/N
cfl = 0.1
hbar_ = 0.01
Mtot = 3e16
ntot = 1e10
A = [10.,10.,10.]
H0 = 6.9e-5 # 70 in normal units
D = a_ini
n_streams = 16#512
expand = 1e-7

DIs = []

def tomlText(Mtot, mft = False):
    if mft:
        return f'''
# all units in kpc, Msolar, Myr
axis_length                 = {L}
final_sim_time              = {T}
cfl                         = {cfl}
num_data_dumps              = {drops}
total_mass                  = {Mtot}
#particle_mass              = None
hbar_                       = {hbar_}
sim_name                    = "{sim_name}-mft"
ntot                        = {ntot}
k2_cutoff                   = 0.95
alias_threshold             = 0.001
dims                        = {len(A)}
size                        = {N}

[ics]                         
type                        = "UserSpecified"
path                        = "{ics_dir}/{sim_name}.npz"

[cosmology]
omega_matter_now            = 1.0
omega_radiation_now         = 0.0
h                           = {expand}
z0                          = {1.0/a_ini  - 1.0}
max_dloga                   = 0.01

[remote_storage_parameters]
keypair = "dev.json"
storage_account = "streams"
'''
    else:
        return f'''
# all units in kpc, Msolar, Myr
axis_length                 = {L}
final_sim_time              = {T}
cfl                         = {cfl}
num_data_dumps              = {drops}
total_mass                  = {Mtot}
#particle_mass              = None
hbar_                       = {hbar_}
sim_name                    = "{sim_name}"
ntot                        = {ntot}
k2_cutoff                   = 0.95
alias_threshold             = 0.001
dims                        = {len(A)}
size                        = {N}

[ics]                         
type                        = "UserSpecified"
path                        = "{ics_dir}/{sim_name}.npz"

[cosmology]
omega_matter_now            = 1.0
omega_radiation_now         = 0.0
h                           = {expand}
z0                          = {1.0/a_ini  - 1.0}
max_dloga                   = 0.01

[sampling]
num_streams = {n_streams}
seeds = "1 to {n_streams}"
scheme = "Wigner"

[remote_storage_parameters]
keypair = "dev.json"
storage_account = "streams"
'''


def saveToml(text, mft = False):
    g = open(f"{sim_name}-streams.bash", "w")
    g.write("cargo run --release -- \\")

    toml_contents = text
    bash_script = f'''
    --toml {toml_dir}/{sim_name}.toml \\'''
    if mft:
        bash_script = f'''
        --toml {toml_dir}/{sim_name}-mft.toml \\'''

    f = None 
    if mft:
        f = open(f"{toml_dir}/{sim_name}-mft.toml", "w")
    else:
        f = open(f"{toml_dir}/{sim_name}.toml", "w")
    f.write(toml_contents)
    f.close()

    g.write(bash_script)

def coords():
    D = a_ini

    q = np.linspace(-L/2., L/2., N)
    x = np.linspace(-L/2., L/2., N)

    ones = np.ones(N)

    xq = q - D*(L/np.pi/2.)*A[0]*np.sin(2*q*np.pi/L)
    yq = q - D*(L/np.pi/2.)*A[1]*np.sin(2*q*np.pi/L)
    zq = q - D*(L/np.pi/2.)*A[1]*np.sin(2*q*np.pi/L)

    Qx_ = np.interp(x, xq, q)
    Qy_ = np.interp(x, yq, q)
    Qz_ = np.interp(x, zq, q)

    Qx = np.einsum("i,j,k->ijk",Qx_,ones,ones)
    Qy = np.einsum("i,j,k->ijk",ones,Qy_,ones)
    Qz = np.einsum("i,j,k->ijk",ones,ones,Qy_)

    return Qx, Qy, Qz

def calc_n(Qx, Qy, Qz):
    n = 1. 
    Q = [Qx, Qy, Qz]

    for i in range(len(Q)):
        n *= 1./( 1. - D*A[i]*np.cos(2*np.pi*Q[i]/L) ) 
    
    return n

def calc_phi(Qx, Qy, Qz):
    D = a_ini
    f = 1.
    H = H0 / a_ini**3

    phi = 0. 
    Q = [Qx, Qy, Qz]
    factor = a_ini**2 * D * f * H

    for i in range(len(Q)):
        phi += factor * ( A[i] * L**2 / (2*np.pi)**2 * np.cos(Q[i]*2*np.pi/L)
        + .5*D*(A[i]*L/(2*np.pi)*np.sin(Q[i]*2*np.pi/L))**2 )
    
    return phi

def getPsi():
    Qx, Qy, Qz = coords()
    n = calc_n(Qx, Qy, Qz)
    phi = calc_phi(Qx, Qy, Qz)
    psi = np.sqrt(n)*np.exp(1j * phi / hbar_)

    Mtot = np.sum(np.abs(psi)**2)*dx

    # plt.imshow( np.abs(psi)**2, interpolation = 'none', 
    #     extent=[-L/2., L/2., -L/2., L/2.], 
    #     aspect = 'auto', origin = 'lower',cmap = 'viridis')
    # plt.show()

    return psi / np.sqrt(Mtot),Mtot

def savePsi(psi):
    np.savez(f"{ics_dir}/{sim_name}.npz", real = psi.real, imag = psi.imag)


def makeFigDir():
    isExist = os.path.exists("figs/" + sim_name + "/")
    if not(isExist):
        os.mkdir("figs/" + sim_name + "/")

    isExist = os.path.exists("sim-data/" + sim_name + "-combined/")
    if not(isExist):
        os.mkdir("sim-data/" + sim_name + "-combined/")


if __name__ == "__main__":
    psi, _ = getPsi()
    savePsi(psi)

    text = tomlText(Mtot)
    saveToml(text)

    text_mft = tomlText(Mtot, True)
    saveToml(text_mft, True)

    # makeFigDir()
    #os.system(f"cargo run --release --features=expanding -- --toml {sim_name}-{N}.toml")
    os.system(f"cargo run --bin msm-simulator --release --no-default-features --features=expanding,remote-storage -- --toml {toml_dir}/{sim_name}.toml")
    os.system(f"cargo run --bin msm-simulator --release --no-default-features --features=expanding,remote-storage -- --toml {toml_dir}/{sim_name}-mft.toml")
    # os.system(f"cargo run --bin msm-synthesizer --release --no-default-features --features=expanding -- --toml {toml_dir}/{sim_name}.toml")

    # for i in range(len(DIs)):
    #     DIs[i].Main(sim_name)
