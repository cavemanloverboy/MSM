import time 
import scipy.fftpack as sp
import numpy as np 
import gravSolver as g
import SimObj as S 
import pyfftw
import utils as u_
import numpy as np 

r = 2
factor = 2**r

ofile = "gravTest" + str(r)
dtype_ = "complex128"
##### configuration parameters #####

N = 128 # number of pixels on each axis
Ns = 2
n = 1024*factor 
L = 60. # length of each axis

dx = L/N # size of individual pixel
dk = 2.*np.pi/L # size of pixel in k space

dt = 10. # time step
Tfinal = 1000. # final time
dataDrops = 10 # how many data dumps I want

Mtot = 1e12 # total mass
C = 4*np.pi*4.65e-12 # poisson constant
hbar = 1.76e-90  # planks constant
mpart = (1e-22)*9.0e-67 # mass of specific particle
hbar_ = hbar/mpart

dIs = []
np.random.seed(1)

def initSolver(s, tag, perturb = True):
    gr = g.Grav3D(s.N, s.Ns)

    gr.psi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
    gr.fft_out = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
    gr.fft_out_phi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)

    gr.T = 0

    k_max = np.max(s.kx_ord)

    gr.psi = gr.ColdGauss_k([0,0,0], [k_max/20.,k_max/20.,k_max/20.], N, L, gr.fft_out, dtype_, 1)
    gr.psi /= np.sqrt((np.abs(gr.psi)**2).sum())

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