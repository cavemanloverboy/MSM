import pyfftw
import numpy as np 
from matplotlib import animation
import time
import datetime
import sys
import os
import IC3D as ic
import scipy.fftpack as sp
from numba import vectorize, complex64#, complex128
import dataInterp as dI
import matplotlib.pyplot as plt 

##### configuration parameters #####

N = 16 # number of pixels on each axis
L = 60. # length of each axis

dx = L/N # size of individual pixel
dk = 2.*np.pi/L # size of pixel in k space

dt = 10. # time step
Tfinal = 500000. # final time
dataDrops = 200 # how many data dumps I want

Mtot = 1e12 # total mass
M_BH_frac = 0#10. # fraction of the black hole mass to the darkmatter mass
C = 4*np.pi*4.65e-12 # poisson constant
hbar = 1.76e-90  # planks constant
mpart = (1e-22)*9.0e-67 # mass of specific particle
hbar_ = hbar/mpart

simName = "testSim"
dtype_ = "complex64"


class sim(object):
	def __init__(self):

		##### physics parameters #####
		self.psi = None # wave function
		self.spec_grid = None # kx**2 + ky**2 + kz**2

		self.fft_out = None # output of fft obj
		self.fft_obj = None # fft object
		self.ifft_obj = None # inverse fft obj

		self.T = None # in sim time

		##### simulation parameters #####
		self.time0 = None # real script start time
		self.start = None # simulation start time, after IC initialization
s = sim()


def remaining(done, total):
	Dt = time.time() - s.start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)//(60*60)
	mins = int(r%(60*60))//(60)
	s = int(r%60)
	return (hrs, mins, s)

def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()


def EndSim():
	f = open(simName + "/" + simName + "Meta.txt", 'a')
	f.write("sim completed: " + str(datetime.datetime.now()) + '\n')
	f.write('completed in %i hrs, %i mins, %i s' %hms(time.time()-s.time0))
	f.close()

	dur1 = .6
	dur2 = .4
	freq1 = 400
	freq2 = 500
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (dur1, freq1))
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (dur2, freq2))
	time.sleep(.1)
	os.system('spd-say "program, %s, completed"'%(simName))

	print('\ncompleted in %i hrs, %i mins, %i s' %hms(time.time()-s.time0))
	print("output: ", simName)


def DropData(i):
	f = open(simName + "/" + simName + "Meta.txt", 'a')
	f.write("drop " + str(i) + "completed: " + str(datetime.datetime.now()) + '\n')
	f.close()

	np.save(simName + "/" + "drop" + str(i) + ".npy", s.psi)



@vectorize([complex64(complex64, complex64)], target = "parallel")
def mult(x,y):
	return x*y


def compute_phi(rho_r):

	fftObj = pyfftw.FFTW(rho_r, s.fft_out_phi, axes = (0,1,2))
	rho_k = fftObj()
	phi_k = None
	with np.errstate(divide ='ignore', invalid='ignore'):
	    # np.errstate(...) suppresses the divide-by-zero warning for k = 0
	    phi_k = -(C*rho_k)/((s.spec_grid))
	phi_k[0,0,0] = 0.0 # set k=0 component to 0
	ifftObj = pyfftw.FFTW(phi_k.astype(np.complex64), s.fft_out_phi, axes = (0,1,2), direction = "FFTW_BACKWARD")
	phi_r = ifftObj()
	return phi_r

def update():

		rho_r = Mtot*(np.abs(s.psi))**2
		rho_r[N//2-1:N//2+1, N//2-1:N//2+1, N//2-1:N//2+1] += M_BH_frac*Mtot/8.
		phi_r = compute_phi(rho_r + 0j).real
		
		Vr = phi_r*mpart

		# update position half-step
		s.psi = mult(s.psi, np.exp(-1j*dt*Vr/(2.*hbar))+0j)
		#s.psi *= np.exp(-1j*dt*Vr/(2.*hbar))

		# update momentum full-step
		fftObj = pyfftw.FFTW(s.psi, s.fft_out, axes = (0,1,2))
		psi_k = fftObj()
		psi_k *= np.exp(-1j*dt*hbar*(s.spec_grid)/(2.*mpart))
		ifftObj = pyfftw.FFTW(psi_k, s.psi, axes = (0,1,2), direction = "FFTW_BACKWARD")
		s.psi = ifftObj()

		# update position half-step
		s.psi = mult(s.psi, np.exp(-1j*dt*Vr/(2.*hbar))+0j)


def RunSim():
	
	tNext = float(Tfinal)/dataDrops
	drop = 0

	while(s.T < Tfinal):
		update()
		if (s.T > tNext):
			tNext += float(Tfinal)/dataDrops
			drop += 1
			DropData(drop)
			repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(drop, dataDrops)))
		s.T += dt
	
	drop += 1
	DropData(drop)


def SetICs():
	s.psi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
	s.fft_out = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
	s.fft_out_phi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)

	s.T = 0

	############ initial conditions ##############
	dx = L/float(N)
	kx = 2*np.pi*sp.fftfreq(N, d = dx)
	k_max = np.max(kx)
	s.psi = ic.ColdGauss_k([0,0,0], [k_max/20.,k_max/20.,k_max/20.], N, L, s.fft_out, dtype_, 1)
	s.psi /= np.sqrt((np.abs(s.psi)**2).sum()*dx)

	s.spec_grid = ic.MakeSpecGrid(N, dx)

	DropData(0)
	s.start = time.time()


def MakeMetaFile(**kwargs):
	f = open(simName + "/" + simName + "Meta.txt", 'w+')
	f.write("sim start: " + str(datetime.datetime.now()) + '\n\n')

	for key, value in kwargs.items():
		f.write(str(key) + ": " + str(value) + '\n')
	f.write('\n')

	f.close()


def PrepareSim():
	s.time0 = time.time()
	try:
		os.mkdir(simName)
	except OSError:
		pass
	MakeMetaFile(N = N, L = L, dt = dt, Mtot = Mtot, C = C, hbar_ = hbar_, mpart = mpart, Tfinal = Tfinal)


def main():
	PrepareSim()
	SetICs()
	RunSim()
	EndSim()
	dI.main(simName)


if __name__ == "__main__":
	main()
