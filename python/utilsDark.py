import pyfftw
from numba import vectorize, complex64, complex128, jit
import scipy.fftpack as sp
import numpy as np
import cupy as cp
import os
import pylab as pyl 
import time
import sys
import matplotlib.pyplot as plt
import shutil

def compute_phi3D(rho_r, s):
	if s.parrallel:
		fftObj = pyfftw.FFTW(rho_r, s.fft_out_phi, axes = (0,1,2),threads = s.threads)
	else:
		fftObj = pyfftw.FFTW(rho_r, s.fft_out_phi, axes = (0,1,2))
	rho_k = fftObj()
	phi_k = None
	with np.errstate(divide ='ignore', invalid='ignore'):
	    # np.errstate(...) suppresses the divide-by-zero warning for k = 0
	    phi_k = -(rho_k)/((s.spec_grid))
	phi_k[0,0,0] = 0.0 # set k=0 component to 0
	if s.parrallel:
		ifftObj = pyfftw.FFTW(phi_k, s.fft_out_phi, axes = (0,1,2), direction = "FFTW_BACKWARD",threads = s.threads)
	else:
		ifftObj = pyfftw.FFTW(phi_k, s.fft_out_phi, axes = (0,1,2), direction = "FFTW_BACKWARD")
	phi_r = ifftObj()
	return phi_r

def compute_phi3D_cp(rho, k):
	rho_k = fft3_cp(rho)
	with np.errstate(divide ='ignore', invalid='ignore'):
		phi_k = -1*(rho_k)/k
	phi_k[0,0,0] = 0.0
	return ifft3_cp(phi_k)

def ifft3_cp(x):
	return cp.fft.ifft(cp.fft.ifft(cp.fft.ifft( x, axis = 0 ), axis = 1), axis = 2)

def fft3_cp(x):
	return cp.fft.fft(cp.fft.fft(cp.fft.fft( x, axis = 0 ), axis = 1), axis = 2)


def MakeSpecGrid(N, dx_ = None):
	if dx_ is None:
		dx_ = 1./N
	kx = 2*np.pi*sp.fftfreq(N,d = dx_)
	ky = 2*np.pi*sp.fftfreq(N,d = dx_)
	kz = 2*np.pi*sp.fftfreq(N,d = dx_)

	ky,kx,kz = np.meshgrid(ky,kx,kz)

	return kx**2 + ky**2 + kz**2

def MakeSpecGrid_cp(N, dx_ = None):
	if dx_ is None:
		dx_ = 1./N
	kx = 2*np.pi*cp.fft.fftfreq(N,d = dx_)
	ky = 2*np.pi*cp.fft.fftfreq(N,d = dx_)
	kz = 2*np.pi*cp.fft.fftfreq(N,d = dx_)

	ky,kx,kz = cp.meshgrid(ky,kx,kz)

	return kx**2 + ky**2 + kz**2

@vectorize([complex128(complex128, complex128)], target = "parallel")
def mult(x,y):
	return x*y

@vectorize([complex128(complex128, complex128)], target = "parallel")
def divide(x,y):
	return x/y

def normalize(y,dx):
	return y/(np.sum(y)*dx)

def remaining(done, total, start):
	Dt = time.time() - start
	return hms((Dt*total)/float(done) - Dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
	r = T
	hrs = int(r)/(60*60)
	mins = int(r%(60*60))/(60)
	s = int(r%60)
	return (hrs, mins, s)

def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()


def getMetaKno(name, tag = "", **kwargs):
	f = open("../" + name + "/" + tag + "Meta.txt")

	print "reading meta info..."
	
	metaParams = {}

	for key_ in kwargs.keys():
		metaParams[key_] = None

	for line in f.readlines():
		for key_ in kwargs.keys():
			if key_ + ":" in line:
				print line
				number = line.split(":")[1]
				#metaParams[key_] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]
				if key_ == "N" or key_ == "frames" or key_ == "n" or key_ == "drops":
					metaParams[key_] = int(number)
				elif key_ == "tags":
					list_ = number.split(",")
					for i in range(len(list_)):
						list_[i] = list_[i].replace("\n","").strip()
					metaParams[key_] = list_
				elif key_ == "dtype_":
					metaParams[key_] = str(number).strip()
				else:
					metaParams[key_] = float(number)

	return metaParams


def orientPlot():
	plt.rc("font", size=22)

	plt.figure(figsize=(6,6))

	fig,ax = plt.subplots(figsize=(6,6))

	plt.rc("text", usetex=True)

	plt.rcParams["axes.linewidth"]  = 1.5

	plt.rcParams["xtick.major.size"]  = 8

	plt.rcParams["xtick.minor.size"]  = 3

	plt.rcParams["ytick.major.size"]  = 8

	plt.rcParams["ytick.minor.size"]  = 3

	plt.rcParams["xtick.direction"]  = "in"

	plt.rcParams["ytick.direction"]  = "in"

	plt.rcParams["legend.frameon"] = 'False'


def GetIndexes(T,N,times):
	inds = []
	for i in range(len(times)):
		t = times[i]
		ind_ = int(t*N/T)
		ind_ = np.min([ind_, N-1])
		ind_ = np.max([0,ind_])
		inds.append(ind_)
	return inds


def readyDir(ofile, tag):
	try: # attempt to make the directory
		os.mkdir("../" + ofile + "/" + tag)
	except OSError:
		try: # assuming directory already exists, delete it and try again
			print "removing and recreating an existing directory: " + "../" + ofile + "/" + tag
			shutil.rmtree("../" + ofile + "/" + tag)
			readyDir(ofile, tag)
		except OSError:
			pass


def ding():
	dur1 = .15
	dur2 = .15
	freq1 = 600
	freq2 = 700
	os.system('play  --no-show-progress --null --channels 1 synth %s sine %f' % (dur1, freq1))
	time.sleep(.04)
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (dur2, freq2))


def setFigObj(name, fo):
	meta = getMetaKno(name, tags = "tags", Tfinal = "Tfinal", drops = "drops")
	
	fo.tags = meta["tags"]
	print meta["tags"]
	fo.Tfinal = meta["Tfinal"]
	fo.drops = meta["drops"]

	fo.N, fo.M_BH_frac, fo.L = [], [], []
	fo.hbar_, fo.chi, fo.dtau = [], [], []
	fo.dtype_, fo.phi_max, fo.phi_min = [], [], []


	for i in range(len(fo.tags)):
		tag_ = fo.tags[i]
		meta = getMetaKno(name, tag_, N = "N", M_BH_frac = "M_BH_frac", L = "L",
			hbar_ = "hbar_", chi = "chi", dtau = "dtau", dtype_ = "dtype_",
			phi_max = "phi_max", phi_min = "phi_min")
		fo.N.append(meta["N"])
		fo.M_BH_frac.append(meta["M_BH_frac"])
		fo.L.append(meta["L"])
		fo.hbar_.append(meta["hbar_"])
		fo.chi.append(meta["chi"])
		fo.dtau.append(meta["dtau"])
		fo.dtype_.append(meta["dtype_"])
		fo.phi_max.append(meta["phi_max"])
		fo.phi_min.append(meta["phi_min"])



# gets the names of files in the given directory organized by time stamp (ascending)
def getNames(name):
	files = ["../" + name + "/" + file for file in os.listdir("../" + name) if (file.lower().endswith('.npy'))]
	files.sort(key=os.path.getmtime)
	return files


def setFileNames(fo,name):
	fo.fileNames = []
	for i in range(len(fo.tags)):
		fo.fileNames.append(getNames(name + "/" + fo.tags[i]))
	fo.N_images = len(fo.fileNames[0])



class figObj(object):

	def __init__(self):
		self.axs = None
		self.ims = None
		self.time_text = None

		self.tags = None
		self.Tfinal = None
		self.drops = None

		self.fileNames = None

		self.N = None
		self.M_BH_frac = None
		self.L = None
		self.hbar_ = None
		self.chi = None
		self.dtau = None
		self.dtype_ = None
		self.phi_max = None
		self.phi_min = None
		self.x = None

		self.N_images = None

		self.currentMax = [0,0]
		self.currentMin = [0,0]

		self.xCuts = None
		self.yCuts = None
		self.zCuts = None
		self.ex = None # current extent

		self.start_ = None

		

