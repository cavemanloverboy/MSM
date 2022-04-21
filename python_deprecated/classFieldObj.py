import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import time
import datetime
import utils as u
import os
import scipy.fftpack as sp


class MS1D(object):

    def __init__(self, N, n_s):

        self.Psi = np.ones((n_s, N)) + 0j

        self.tag = ''

        self.working = True 

        self.T_scale = 1.

        self.kord = np.arange(N) # momentum modes

        self.n = 1.


    def ReadyDir(self,ofile):
        try:
            os.mkdir("../Data/" + ofile + "/Psi" + self.tag)
        except OSError:
            pass

    def DataDrop(self,i,ofile_):
        np.save("../Data/" + ofile_ + "/Psi" + self.tag + "/" + "drop" + str(i) + ".npy", self.Psi)

    def getRhoX(self,s):
        #return s.Mtot*(np.abs(self.Psi)**2).mean(axis = 0)
        return s.Mtot*np.abs(self.Psi)**2

    def getRhoK(self, s):
        Psi_k = sp.fft(self.Psi) / np.sqrt(s.N)
        return s.Mtot*(np.abs(Psi_k)**2).mean(axis = 0)

    def getPotential(self,s):
        rho_r = self.getRhoX(s)
        rho_k = sp.fft(rho_r) 
        phi_k = None 
        with np.errstate(divide ='ignore', invalid='ignore'):
            # np.errstate(...) suppresses the divide-by-zero warning for k = 0
            phi_k = -(s.C*rho_k)/(s.kx**2)
        phi_k[:,0] = 0
        phi_r = np.real(sp.ifft(phi_k))
        return phi_r

    def Update(self, dt_, s):

        for i in range(s.framesteps):    
            # update position half-step
            psi_k = sp.fft(self.Psi)
            psi_k *= np.exp(-1j*dt_*self.T_scale*s.hbar*(s.kx**2)/(4*s.mpart))
            self.Psi = sp.ifft(psi_k)

            phi_r = self.getPotential(s)
            Vr = phi_r*s.mpart # the PE field for an individual particle

            # update momentum full-step
            self.Psi *= np.exp(-1j*dt_*self.T_scale*Vr/(s.hbar))

            # update position half-step
            psi_k = sp.fft(self.Psi)
            psi_k *= np.exp(-1j*dt_*self.T_scale*s.hbar*(s.kx**2)/(4*s.mpart))
            self.Psi = sp.ifft(psi_k)


    def readyDir(self,ofile):
        os.mkdir("../Data/" + ofile + "/Psi" + self.tag)

    def dropData(self,i,ofile_):
        np.save("../Data/" + ofile_ + "/Psi" + self.tag + "/" + "drop" + str(i) + ".npy", self.Psi)