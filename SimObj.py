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


class SimObj(object):

    def __init__(self):
        s = self 

        # simulation parameters
        s.N = None
        s.L = None 
        s.dt = None 
        s.Mtot = None 
        s.C = None 
        s.Lambda0 = 0. 
        s.hbar_ = None 
        s.mpart = None 
        s.hbar = None 
        s.Tfinal = None
        s.Ns = None 
        s.n = None 
        s.T = None 
        s.spec_grid = None  
        
        # meta params
        s.time0 = None
        s.ofile = None
        s.solvers = []
    


    def MakeMetaFile(self, **kwargs):
        s = self
        if not(os.path.isdir("../Data/")):
            os.mkdir("../Data/")
        if not(os.path.isdir("../Data/"+ s.ofile + "/")):
            os.mkdir("../Data/" + s.ofile + "/")

        try:
            os.mkdir("../Data/" + self.ofile + "/")
        except OSError:
            pass
        f = open("../Data/" + self.ofile + "/" + self.ofile + "Meta.txt", 'w+')
        f.write("sim start: " + str(datetime.datetime.now()) + '\n\n')

        for key, value in kwargs.items():
            f.write(str(key) + ": " + str(value) + '\n')
        f.write('\n')

        f.close()

    def MakeSpecGrid(self, N, dx_):
        kx = 2*np.pi*sp.fftfreq(N,d = dx_)
        ky = 2*np.pi*sp.fftfreq(N,d = dx_)
        kz = 2*np.pi*sp.fftfreq(N,d = dx_)

        ky,kx,kz = np.meshgrid(ky,kx,kz)

        self.spec_grid = kx**2 + ky**2 + kz**2


    # runs the simulation
    def Run(self, verbose = True):

        for solver_ in self.solvers:
            solver_.DataDrop(0, self.ofile)

        # loops through data drops
        for i in range(self.frames):
            
            # update the simulation time
            dt_ = self.dt
            self.T += dt_ * self.framesteps

            # loop through each solver
            for solver_ in self.solvers: 
                if solver_.working: # check if solver has broken

                    # ------------ step 3d ---------------- #
                    # update dynamical variables
                    with np.errstate(divide ='ignore', invalid='ignore'):
                        solver_.Update(dt_ * solver_.T_scale, self)   
                    # ------------------------------------- #
                
                if solver_.working:
                    # ------------ step 3e ---------------- #
                    # output state of dynamical variables
                    solver_.DataDrop(i+1, self.ofile)
                    # ------------------------------------- #

            # info drop on terminal
            if verbose:
                u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i + 1, self.frames, self.time0)))


    def ValidIndex(self, i):
        return (i < self.N) and (i >= 0)

    def GetC(self, i, j):
        C_ij = 0.
        
        if (i != j):
            C_ij = self.C / (self.kord[i] - self.kord[j])**2 + self.Lambda0
        else:
            C_ij = self.Lambda0

        return C_ij

    def GetLam(self, ind1, ind2, ind3, ind4):
        C_ij = self.Lambda0
        
        if ind1 != ind3 and np.abs(self.C) > 0:
            C_ij += .5*self.C / (self.kord[ind1] - self.kord[ind3])**2
        if ind1 != ind4 and np.abs(self.C) > 0:
            C_ij += .5*self.C / (self.kord[ind1] - self.kord[ind4])**2

        return C_ij

    def EndSim(self, text = True):
        for solver_ in self.solvers:
            # output state of dynamical variables
            solver_.DataDrop(self.frames, self.ofile)
        if text:
            print('\nsim completed in %i hrs, %i mins, %i s' %u.hms(time.time()-self.time0))
            print("output: ", self.ofile)
