import time 
import scipy.fftpack as sp
import numpy as np 
import gravSolver as g
import SimObj as S 
import pyfftw

r = 2
factor = 2**r

ofile = "gravTest" + str(r)
dtype_ = "complex128"
##### configuration parameters #####

N = 128 # number of pixels on each axis
Ns = 512
n = 1024*factor 
L = 60. # length of each axis

dx = L/N # size of individual pixel
dk = 2.*np.pi/L # size of pixel in k space

dt = 10. # time step
Tfinal = 500000. # final time
dataDrops = 200 # how many data dumps I want

Mtot = 1e12 # total mass
C = 4*np.pi*4.65e-12 # poisson constant
hbar = 1.76e-90  # planks constant
mpart = (1e-22)*9.0e-67 # mass of specific particle
hbar_ = hbar/mpart

np.random.seed(1)

def initSolver(s, peturb = True):
    gr = g.Grav3D(s.N, s.Ns)

    gr.psi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
    gr.fft_out = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
    gr.fft_out_phi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)

    gr.T = 0

    k_max = np.max(s.kx_ord)

    gr.psi = gr.ColdGauss_k([0,0,0], [k_max/20.,k_max/20.,k_max/20.], N, L, s.fft_out, dtype_, 1)
    gr.psi /= np.sqrt((np.abs(s.psi)**2).sum()*dx)



    return gr


# creates the simulation object
def initSim():
    print("Initializing simultion object")
    s = S.SimObj()

    # TODO: set simulation params
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
    x = dx*(.5+np.arange(-N/2, N/2))
    kx = 2*np.pi*sp.fftfreq(N, d = L/N)
    kx_ord = sp.fftshift(kx)
    y = sp.fftfreq(len(kx_ord), d=dk) *N

    s.MakeSpecGrid(s.N, dx)

    # set meta params 
    s.time0 = time.time()
    s.ofile = ofile
    s.MakeMetaFile(N = N, L = L, dt = dt, Mtot = Mtot, C = C, 
    hbar_ = hbar_, mpart = mpart, Tfinal = Tfinal)

    return s


def main(ofile):
    time0 = time.time()

    s = initSim() # initialize the simulation object
    gr = initSolver(s)

if __name__ == "__main__":
	main(ofile)
