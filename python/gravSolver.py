import pyfftw
import numpy as np
import numpy.random as rand
import os
import scipy.fftpack as sp


class Grav3D(object):

    def __init__(self, N, n_s):
        s = self

        self.psi = None 
        self.fft_out = None 
        self.fft_out_phi = None

        self.tag = ''

        self.T = 0.

        self.T_scale = 1.

        self.working = True

    def ColdGauss(self, m, s, N, L, dtype_ = 'complex64'):
        psiX = pyfftw.empty_aligned(N, dtype = dtype_)
        psiY = pyfftw.empty_aligned(N, dtype = dtype_)
        psiZ = pyfftw.empty_aligned(N, dtype = dtype_)

        dx = L/float(N)
        x = dx*(.5+np.arange(-N/2,N/2))

        psiX[:] = np.exp(-.5*((x-m[0])/(s[0]))**2) + 0j
        psiX /= np.sqrt(np.sum((np.abs(psiX)**2))*dx)

        psiY[:] = np.exp(-.5*((x-m[1])/(s[1]))**2) + 0j
        psiY /= np.sqrt(np.sum((np.abs(psiY)**2))*dx)

        psiZ[:] = np.exp(-.5*((x-m[0])/(s[0]))**2) + 0j
        psiZ /= np.sqrt(np.sum((np.abs(psiZ)**2))*dx)

        return psiX[:,np.newaxis,np.newaxis]*psiY[:,np.newaxis]*psiZ


    def ColdGauss_k(self, m, s, N, L, fft_out, dtype_ = 'complex64', rand_ = 0):
        psiX = pyfftw.empty_aligned(N, dtype = dtype_)
        psiY = pyfftw.empty_aligned(N, dtype = dtype_)
        psiZ = pyfftw.empty_aligned(N, dtype = dtype_)

        dx = L/float(N)
        kx = 2*np.pi*sp.fftfreq(N, d = dx)
        dk = 2.*np.pi/L # kpc^-1

        psiX[:] = np.exp(-.5*((kx-m[0])/(s[0]))**2) + 0j
        psiX /= np.sqrt(np.sum((np.abs(psiX)**2))*dk)

        psiY[:] = np.exp(-.5*((kx-m[1])/(s[1]))**2) + 0j
        psiY /= np.sqrt(np.sum((np.abs(psiY)**2))*dk)

        psiZ[:] = np.exp(-.5*((kx-m[0])/(s[0]))**2) + 0j
        psiZ /= np.sqrt(np.sum((np.abs(psiZ)**2))*dk)

        rval = psiX[:,np.newaxis,np.newaxis]*psiY[:,np.newaxis]*psiZ

        rval *= np.exp(1j*rand.uniform(0,np.pi*2.,(N,N,N)))

        ifftObj = pyfftw.FFTW(rval, fft_out, axes = (0,1,2), direction = "FFTW_BACKWARD")
        return ifftObj()

    def ReadyDir(self,ofile):
        if not( os.path.isdir("../Data/" + ofile + "/Psi" + self.tag) ):
            os.mkdir("../Data/" + ofile + "/Psi" + self.tag)

    def DataDrop(self,i,ofile_):
        np.save("../Data/" + ofile_ + "/Psi" + self.tag + "/" + "drop" + str(i) + ".npy", self.psi)

    def compute_phi(self, rho_r, s):

        fftObj = pyfftw.FFTW(rho_r, self.fft_out_phi, axes = (0,1,2))
        rho_k = fftObj()
        phi_k = None
        with np.errstate(divide ='ignore', invalid='ignore'):
            # np.errstate(...) suppresses the divide-by-zero warning for k = 0
            phi_k = -(s.C*rho_k)/((s.spec_grid))
        phi_k[0,0,0] = 0.0 # set k=0 component to 0
        ifftObj = pyfftw.FFTW(phi_k, self.fft_out_phi, axes = (0,1,2), direction = "FFTW_BACKWARD")
        phi_r = ifftObj()
        return phi_r

    def Update(self, dt, Tgoal, s):

        while(self.T < Tgoal):
            dt_ = np.min( [dt, Tgoal - self.T] )

            rho_r = s.Mtot*(np.abs(self.psi))**2
            phi_r = self.compute_phi(rho_r + 0j, s).real

            Vr = phi_r*s.mpart

            # update position full-step
            fftObj = pyfftw.FFTW(self.psi, self.fft_out, axes = (0,1,2))
            psi_k = fftObj()
            psi_k *= np.exp(-1j*dt_*s.hbar*(s.spec_grid)/(4.*s.mpart))
            ifftObj = pyfftw.FFTW(psi_k, self.psi, axes = (0,1,2), direction = "FFTW_BACKWARD")
            self.psi = ifftObj()

            # update momentum full-step
            self.psi *= np.exp(-1j*dt_*Vr/(s.hbar))

            # update momentum full-step
            fftObj = pyfftw.FFTW(self.psi, self.fft_out, axes = (0,1,2))
            psi_k = fftObj()
            psi_k *= np.exp(-1j*dt_*s.hbar*(s.spec_grid)/(4.*s.mpart))
            ifftObj = pyfftw.FFTW(psi_k, self.psi, axes = (0,1,2), direction = "FFTW_BACKWARD")
            self.psi = ifftObj()

            self.T += dt_
