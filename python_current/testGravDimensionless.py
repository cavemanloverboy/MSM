import utilsDark as u 
import SP3D_object as g 
import scipy.fftpack as sp
import Sim_object as Sim 
import time
import pyfftw
import numpy as np
import cupy as cp

r = 2
factor = 2**r

ofile = "gravTest" + str(r)
dtype_ = "complex128"
##### configuration parameters #####

N = 128 # number of pixels on each axis
Ns = 2 # number of streams
n = 1024*factor # number of particles

drops = 100

chi = 1e5
dtau = 1e-7

Tfinal = 1.

N = 256


dIs = []
np.random.seed(1)

def initSolver(s, tag, perturb = True):
    #gr = g.Grav3D(s.N, s.Ns)
    gr = g.SP3D_cp(N = N, chi = chi, dtau = dtau, parrallel = False)

    gr.Tkachev()

    gr.tag = tag

    if perturb:
        dq = np.random.normal(0, np.sqrt(.5), N)
        dp = np.random.normal(0, np.sqrt(.5), N)
        gr.psi += np.sqrt(.5/n)*(dq + 1j*dp)
        gr.psi /= np.sqrt(dx)

    gr.ReadyDir(ofile)

    return gr


# creates the simulation object
def initSim():
    print("Initializing simultion object")
    s = S.SimObj()

    # set simulation params
    s.N = N
    s.Ns = Ns
    s.n = n
    s.L = L
    s.dt = dt
    s.Mtot = Mtot 
    s.C = C 
    s.Lambda0 = 0. 
    s.hbar_ = hbar_ 
    s.mpart = mpart
    s.hbar = hbar
    s.Tfinal = Tfinal
    s.T = 0

    dk = 1./dx
    x = dx*(.5+np.arange(-N//2, N//2))
    kx = 2*np.pi*sp.fftfreq(N, d = L/N)
    kx_ord = sp.fftshift(kx)
    y = sp.fftfreq(len(kx_ord), d=dk) *N

    s.kx_ord = kx_ord
    s.kx = kx
    s.k_cutoff_frac = .99
    s.MakeSpecGrid(s.N, dx)

    # set meta params 
    s.time0 = time.time()
    s.ofile = ofile
    s.drops = dataDrops
    s.MakeMetaFile(N = N, L = L, dt = dt, Mtot = Mtot, C = C, 
    hbar_ = hbar_, mpart = mpart, Tfinal = Tfinal, drops = dataDrops+1)

    return s


def main(ofile):
    time0 = time.time()

    s = initSim() # initialize the simulation object
    for i in range(Ns):
        s.solvers.append(initSolver(s, str(i)))

    s.time0 = time.time()
    s.Run()
    s.EndSim()
    
    time0 = time.time()

    if len(dIs) >= 1:
        print("begining data interpretation")

        for j in range(len(dIs)):
            dIs[j].main(ofile)
        print('analysis completed in %i hrs, %i mins, %i s' %u_.hms(time.time()-time0))
    u_.ding()

if __name__ == "__main__":
	main(ofile)