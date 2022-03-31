import utilsDark as u
import pyfftw
from numba import vectorize, complex64, complex128
import numpy as np
import cupy as cp 

class SP3D(object):

	def __init__(self, psi = False, N = 128, M_BH_frac = 0., L = 1., threads = None,
		hbar_ = 1., chi = 1e5, dtau = .01, phi_max = .05, phi_min = .001, dtype_ = "complex128",
		tag = "psi", parrallel = True):

		self.N = N # grid cells

		if psi:
			self.psi = psi # classical field
		else:
			self.psi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)

		self.T = 0 # internal time

		self.fft_out = pyfftw.empty_aligned((N,N,N), dtype = dtype_)
		self.fft_out_phi = pyfftw.empty_aligned((N,N,N), dtype = dtype_)


		self.M_BH_frac = M_BH_frac

		self.hbar_ = hbar_

		self.L = L
		self.hbar_ = hbar_

		self.dtype_ = dtype_

		self.dx = L/N
		self.dV = self.dx**3

		self.chi = chi

		self.dtau = .01

		if threads is None:
			from multiprocessing import cpu_count
			self.threads = cpu_count()
		else:
			self.threads = threads

		self.spec_grid = u.MakeSpecGrid(N)

		self.phi_max = phi_max
		self.phi_min = phi_min

		self.tag = tag

		self.parrallel = parrallel

		if threads is None:
			from multiprocessing import cpu_count
			self.threads = cpu_count()
		else:
			self.threads = threads


	def getRhoX(self):
		return (np.abs(self.psi))**2

	def getNorm(self):
		return np.sqrt((np.abs(self.psi)**2).sum()*self.dV)

	def getPotential(self):
		rho_r = self.getRhoX()
		N = self.N
		rho_r[N/2-1:N/2+1, N/2-1:N/2+1, N/2-1:N/2+1] += self.M_BH_frac/(8.*self.dV)
		return u.compute_phi3D(rho_r + 0j, self)

	def Tkachev(self, delta_k = None):
		if delta_k is None:
			delta_k = (.05*np.sqrt(np.max(self.spec_grid)))**2
		phase = np.random.uniform(0., 2.*np.pi, (self.N, self.N, self.N))
		psi_k = np.exp(-self.spec_grid/delta_k)*np.exp(phase*1j)
		ifftObj = None
		if self.parrallel:
			ifftObj = pyfftw.FFTW(psi_k, self.psi, axes = (0,1,2), direction = "FFTW_BACKWARD",threads = self.threads)
		else:
			ifftObj = pyfftw.FFTW(psi_k, self.psi, axes = (0,1,2), direction = "FFTW_BACKWARD")
		self.psi = ifftObj()
		self.psi /= self.getNorm()


	def drift(self, dtau):
		s = self
		if s.parrallel:
			fftObj = pyfftw.FFTW(s.psi, s.fft_out, axes = (0,1,2),threads = s.threads)
		else:
			fftObj = pyfftw.FFTW(s.psi, s.fft_out, axes = (0,1,2))
		psi_k = fftObj()
		if s.parrallel:
			psi_k = u.mult(psi_k, np.exp(-1j*dtau*(s.spec_grid)/(2.)))
			ifftObj = pyfftw.FFTW(psi_k, s.psi, axes = (0,1,2), direction = "FFTW_BACKWARD",threads = s.threads)
		else:
			psi_k *= np.exp(-1j*dtau*(s.spec_grid)/(2.))
			ifftObj = pyfftw.FFTW(psi_k, s.psi, axes = (0,1,2), direction = "FFTW_BACKWARD")
		s.psi = ifftObj()

	def kick(self, dtau):
		s = self
		Vr = s.getPotential()
		if s.parrallel:
			s.psi = u.mult(s.psi, np.exp(-1j*Vr*dtau*s.chi))
		else:
			s.psi *= np.exp(-1j*Vr*dtau*s.chi)

	def adjust_dtau(self):
		max_phi_T = self.dtau*np.max((self.spec_grid))/2.
		max_phi_V = np.abs(self.dtau*np.max(self.getPotential()))
		limiting_phi = np.max([max_phi_T, max_phi_V])
		dtau_ = np.max([1., self.phi_min/limiting_phi])*self.dtau
		dtau_ = self.dtau*np.min([1., self.phi_max/limiting_phi])
		self.dtau = dtau_

	def checkAlias(self, k_max = None, m_max = None):
		s = self
		if s.parrallel:
			fftObj = pyfftw.FFTW(s.psi, s.fft_out, axes = (0,1,2),threads = s.threads)
		else:
			fftObj = pyfftw.FFTW(s.psi, s.fft_out, axes = (0,1,2))
		psi_k = fftObj()
		rho_k = np.abs(psi_k)**2 
		rho_k = u.normalize(rho_k, 1.)
		k = np.sqrt(s.spec_grid)
		if k_max is None:
			k_max = np.max(np.abs(k))*.8
		if m_max is None:
			m_max = .05
		if rho_k[k>k_max].sum() > m_max:
			return "alias"
		else:
			return False

	def readyDir(self,ofile_sim):
		u.readyDir(ofile_sim, self.tag)

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/" + self.tag + "/" + "drop" + str(i) + ".npy", self.psi)

	def update(self, T_f):

		self.adjust_dtau()

		while(self.T < T_f):
			dtau = np.min([self.dtau, T_f - self.T])
			self.drift(dtau/2.)
			self.kick(dtau)
			self.drift(dtau/2.)
			self.T += dtau

		return self.checkAlias()



class SP3D_cp(object):

	def __init__(self, psi = False, N = 128, M_BH_frac = 0., L = 1., threads = None,
		hbar_ = 1., chi = 1e5, dtau = .01, phi_max = .05, phi_min = .001, dtype_ = "complex128",
		tag = "psi", parrallel = True):

		self.N = N # grid cells

		if psi:
			self.psi = psi # classical field
		else:
			self.psi = np.ones((N,N,N))
			self.psi = cp.asarray(self.psi)

		self.T = 0 # internal time

		self.M_BH_frac = M_BH_frac

		self.hbar_ = hbar_

		self.L = L
		self.hbar_ = hbar_

		self.dtype_ = dtype_

		self.dx = L/N
		self.dV = self.dx**3

		self.chi = chi

		self.dtau = .01

		if threads is None:
			from multiprocessing import cpu_count
			self.threads = cpu_count()
		else:
			self.threads = threads

		self.spec_grid = cp.asarray(u.MakeSpecGrid(N))

		self.phi_max = phi_max
		self.phi_min = phi_min

		self.tag = tag

		self.k_check = cp.sqrt(cp.max((self.spec_grid)))
		self.k_max = self.k_check*1.


	def getRhoX(self):
		return (cp.abs(self.psi))**2

	def getNorm(self):
		return cp.sqrt((cp.abs(self.psi)**2).sum()*self.dV)

	def getPotential(self):
		rho_r = self.getRhoX()
		N = self.N
		rho_r[N/2-1:N/2+1, N/2-1:N/2+1, N/2-1:N/2+1] += self.M_BH_frac/(8.*self.dV)
		return u.compute_phi3D_cp(rho_r, self.spec_grid)

	def Tkachev(self, dk_ = .05, delta_k = None):
		if delta_k is None:
			delta_k = (dk_*cp.sqrt(cp.max(self.spec_grid)))**2
		phase = cp.random.uniform(0., 2.*np.pi, (self.N, self.N, self.N))
		psi_k = cp.exp(-self.spec_grid/delta_k)*cp.exp(phase*1j)
		self.psi = u.ifft3_cp(psi_k)
		self.psi /= self.getNorm()


	def drift(self, dtau):
		s = self
		psi_k = u.fft3_cp(s.psi)
		psi_k *= cp.exp(-1j*dtau*(s.spec_grid)/(2.))
		s.psi = u.ifft3_cp(psi_k)

	def kick(self, dtau):
		s = self
		Vr = s.getPotential()
		s.psi *= cp.exp(-1j*Vr*dtau*s.chi)

	def getMaxK(self, prev_k_worked = False):
		if self.checkAlias(self.k_check):
			if prev_k_worked:
				self.k_check *= 2.
				return np.min([self.k_check, self.k_max])
			else:
				self.k_check *= 2.
				return self.getMaxK()
		else:
			self.k_check /= 2.
			return self.getMaxK(True)


	def adjust_dtau(self):
		k_ = self.getMaxK()
		#max_phi_T = self.dtau*cp.max((self.spec_grid))/2.
		max_phi_T = self.dtau*(k_**2)/2.
		max_phi_V = cp.abs(self.dtau*cp.max(self.getPotential()))
		limiting_phi = np.max([max_phi_T, max_phi_V])
		dtau_ = np.max([1., self.phi_min/limiting_phi])*self.dtau
		dtau_ = self.dtau*np.min([1., self.phi_max/limiting_phi])
		self.dtau = dtau_

	def checkAlias(self, k_max = None, m_max = None):
		s = self
		psi_k = u.fft3_cp(s.psi)
		rho_k = cp.abs(psi_k)**2 
		rho_k = u.normalize(rho_k, 1.)
		k = cp.sqrt(s.spec_grid)
		if k_max is None:
			k_max = cp.max(cp.abs(k))*.8
		if m_max is None:
			m_max = .05
		if rho_k[k>k_max].sum() > m_max:
			return "alias"
		else:
			return False

	def readyDir(self,ofile_sim):
		u.readyDir(ofile_sim, self.tag)

	def dropData(self,i,ofile_):
		np.save("../" + ofile_ + "/" + self.tag + "/" + "drop" + str(i) + ".npy", cp.asnumpy(self.psi))

	def update(self, T_f):

		self.adjust_dtau()

		while(self.T < T_f):
			dtau = np.min([self.dtau, T_f - self.T])
			self.drift(dtau/2.)
			self.kick(dtau)
			self.drift(dtau/2.)
			self.T += dtau

		return self.checkAlias()



