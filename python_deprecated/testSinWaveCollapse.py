import SimObj as S 
import classFieldObj as cl 
import time 
import numpy as np
import utils as u
import scipy.fftpack as sp
import matplotlib.pyplot as plt
import di_multiPsi

r = 10
ofile = 'SinWaveH_r' + str(r)
frames = 600
#framesteps = 100#800 * 2
framesteps = 400#800 * 2
dt = 1e-6

N = 256
factor = 2**r
n = int(1024*factor) # number of particles in simulation
Ns = 1024#512 # number of streams in the simulation

L = 1. 
dx = L/N

Mtot = 1.
mpart = Mtot/n
#hbar_ = 1.0e-6
hbar_ = 2.5e-4
#C = .000002
C = .1

omega0S = np.sqrt(np.abs(C)*Mtot/L)
T_scaleS = np.pi/omega0S
#T_scaleS = 10.

L = 1.
dx = L/N # kpc, size of a single pixel

tau = dt*frames*framesteps*hbar_/L**2
chi = Mtot*C*L**3/hbar_**2
print tau, tau*chi

# sampling schemes:
#   - H: Husimi
#   - P: Poisson
#   - W: Wigner
samplingScheme = 'H'

dIs = [di_multiPsi] 


def initSim():
    print("Initializing simultion object")
    s = S.SimObj()

    s.N = N
    s.dt = dt 
    s.hbar = hbar_*mpart
    s.mpart = mpart
    s.C = C
    s.Mtot = Mtot
    s.frames = frames
    s.framesteps = framesteps
    s.ofile = ofile

    dk = 1./dx
    x = dx*(.5+np.arange(-N/2, N/2))
    kx = 2*np.pi*sp.fftfreq(N, d = L/N)
    kx_ord = sp.fftshift(kx)
    y = sp.fftfreq(len(kx_ord), d=dk) *N

    #s.kord = np.arange(N) - N/2
    #s.kx = sp.fftshift(s.kord)
    s.kord = kx_ord
    s.kx = kx
    kmax = np.max(np.abs(s.kord))
    s.L = 1.
    s.dx = s.L/s.N 

    s.x = x
    s.dk = 1./s.dx

    s.Norm = np.sqrt(n)

    s.MakeMetaFile(N = N, dt = dt, frames = frames, framesteps = framesteps,
        C = C, hbar_ = hbar_, Norm = s.Norm, T_scale = T_scaleS, L = L,
        Mtot = Mtot, samplingScheme = samplingScheme)

    return s


def initcl(s):
    x = dx*(.5+np.arange(-N/2, N/2))
    rho = 1. + .1*np.cos(x*2*np.pi)
    psi = np.sqrt(rho)
    psi /= np.sqrt(np.sum(np.abs(psi)**2)) # |psi(x)|**2 = n(x)/n_tot
    #psi *= np.sqrt(n) # |psi(x)|**2 = n(x)
    
    M = len(psi)

    psi_k = sp.fft(psi)

    cl_ = cl.MS1D(N, Ns)
    cl_oneField = cl.MS1D(N,1)
    cl_oneField.tag = '1F'

    cl_.Psi = np.ones((Ns, N)) + 0j
    cl_oneField.Psi = np.ones((1, N)) + 0j

    np.random.seed(1)

    # |psi(x)|**2 = n(x)/n_tot
    for i in range(Ns):
        if samplingScheme.lower() == 'p':
            # Poisson line
            psi_ = psi*np.sqrt(n)
            psi_ = np.sqrt(np.random.poisson( np.abs(psi_**2) ) )  + 0j
            psi_ *= np.exp(1j*np.angle(psi))
            psi_ /= np.sqrt(n)
        elif samplingScheme.lower() == 'w':
            # wigner function
            dq = np.random.normal(0, np.sqrt(.5), N)
            dp = np.random.normal(0, np.sqrt(.5), N)
            psi_ = psi + np.sqrt(.5/n)*(dq + 1j*dp)
        else:
            # husimi function
            dq = np.random.normal(0, np.sqrt(.5), N)*np.sqrt(2)
            dp = np.random.normal(0, np.sqrt(.5), N)*np.sqrt(2)
            psi_ = psi + np.sqrt(.5/n)*(dq + 1j*dp)
        cl_.Psi[i,:] = psi_/np.sqrt(dx) # |psi(x)|**2 = n(x)/n_tot now its densities
    cl_oneField.Psi[0,:] = psi/np.sqrt(dx)

    # norm checks all have appropriate stats
    #cl_.Psi = cl_.Psi*np.sqrt(dx*n)
    #P = (cl_.Psi - np.conj(cl_.Psi))*1j/np.sqrt(2)
    #Q = (cl_.Psi + np.conj(cl_.Psi))/np.sqrt(2)
    #N_ = np.abs(cl_.Psi)**2
    #print N_.mean(axis = 0).sum()/n, n
    #print np.mean(N_.sum(axis = 1)), np.var(N_.sum(axis = 1))
    #assert(0)

    cl_.T_scale = T_scaleS
    cl_oneField.T_scale = T_scaleS

    cl_.kord = s.kord
    cl_oneField.kord = s.kord

    cl_.ReadyDir(ofile)
    cl_oneField.ReadyDir(ofile)

    return cl_, cl_oneField



def main():
    time0 = time.time()

    s = initSim() # initialize the simulation object
    cl_, cl_oneField = initcl(s)
    s.solvers = [ cl_, cl_oneField ]

    print 'initialization completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    print "begining sim", ofile

    #cl_.Update(s.dt, s)
    #plt.plot(s.kx, cl_.getRhoK(s))
    #plt.show()
    #assert(0)

    s.time0 = time.time()
    s.Run()
    s.EndSim()
    
    time0 = time.time()

    if len(dIs) >= 1:
        print "begining data interpretation"

        for j in range(len(dIs)):
            dIs[j].main(ofile)
        print 'analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
    u.ding()



if __name__ == "__main__":
    main()