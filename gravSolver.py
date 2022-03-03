import pyfftw
import numpy as np
import numpy.random as rand
import scipy.fftpack as sp


class Grav3D(object):

    def __init__(self, N, n_s):
        s = self

        self.psi = None 
        self.fft_out = None 
        self.fft_out_phi = None

        self.T = 0.

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