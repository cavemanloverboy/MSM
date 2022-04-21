import utils as u
import numpy as np
import QUtils as qu 
from matplotlib import animation
import multiprocessing as mp 
import scipy.fftpack as sp
import time
import matplotlib.pyplot as plt
import pickle 
from numpy import linalg as LA 
import scipy.stats as st 

simName = 'SinWave_r1'

class figObj(object):

    def __init__(self):
        self.meta = None

        self.fileNames_Psi = None 

        self.N = None 
        self.L = 1.
        self.ntot = None
        self.dt = None 
        self.hbar_ = None
        self.Mtot = None 
        self.framesteps = None 
        self.Norm = None
        self.C = None 
        self.n = None
        self.T_scale = None

        self.dx = None

        self.name = None 
        self.tags = None
        
        self.psi = None 
        self.psi_cl = None 
        self.a = None 
        self.a_cl = None 
        self.N_x = None 
        self.N_k = None 
        self.t = None
fo = figObj()

def setFigObj(name):
    # read in simulation parameters
    meta = u.getMetaKno(name, dir = 'Data/', N = "N", dt = "dt", frames = "frames",
        framesteps = "framesteps", Norm = "Norm", hbar_ = "hbar_", T_scale = 'T_scale',
        C = 'C', n = 'n', Mtot = 'Mtot', L = 'L')
    fo.meta = meta

    fo.name = name
    fo.L = fo.meta['L']
    fo.Mtot = fo.meta['Mtot']
    fo.N = fo.meta['N']
    fo.dx = fo.L/fo.N
    fo.dt = fo.meta["dt"]
    fo.framesteps = fo.meta["framesteps"]
    fo.Norm =  fo.meta["Norm"]
    fo.frames = fo.meta['frames']
    fo.hbar_ = fo.meta['hbar_']
    fo.Norm = fo.meta['Norm']
    fo.C = fo.meta['C']
    fo.T_scale = fo.meta['T_scale']
    fo.n = fo.Norm**2


def SaveStuff(t, Num_x, Num_k, M, eigs, aa, a, Q, psi,psi_cl, label=''):
    np.save("../Data/" + fo.name + "/" + "_t" + label + ".npy", t)
    np.save("../Data/" + fo.name + "/" + "_N_x" + label + ".npy", Num_x)
    np.save("../Data/" + fo.name + "/" + "_N_k" + label + ".npy", Num_k)
    np.save("../Data/" + fo.name + "/" + "_a" + label + ".npy", a)
    np.save("../Data/" + fo.name + "/" + "_psi_x" + label + ".npy", psi)
    np.save("../Data/" + fo.name + "/" + "_psi_x_cl" + label + ".npy", psi_cl)
    np.save("../Data/" + fo.name + "/" + "_Q" + label + ".npy", Q)

def analyze(tag = ''):
    fileNames_Psi = u.getNamesInds("Data/" + fo.name + "/" + "Psi" + tag)
    fo.fileNames_Psi = fileNames_Psi
    N_files = len(fileNames_Psi)
    
    t = np.zeros(N_files)
    a = np.zeros( (N_files, fo.N) ) + 0j
    psi = np.zeros( (N_files, fo.N) ) + 0j
    N_k = np.zeros( (N_files, fo.N) )
    N_x = np.zeros( (N_files, fo.N) )
    M = np.zeros( (N_files,  fo.N, fo.N) ) + 0j
    eigs = np.zeros( (N_files, fo.N) ) + 0j
    aa = np.zeros( (N_files, fo.N, fo.N) ) + 0j 
    Q = np.zeros( N_files )

    for i in range(N_files):
        t_ = i*fo.dt*fo.framesteps*fo.T_scale
        t[i] = t_ 

        # Psi_[stream index][spatial index]
        Psi_ = np.load(fileNames_Psi[i])*np.sqrt(fo.dx*fo.n)
        Psi_k = sp.fft(Psi_) / np.sqrt(fo.N)

        a_ = np.mean(Psi_k, axis = 0)
        a[i,:] = a_

        psi[i,:] = np.mean(Psi_, axis = 0)
        N_x[i,:] = np.mean(np.abs(Psi_)**2, axis = 0)
        
        N_k_ = np.mean(np.abs(Psi_k)**2, axis = 0)
        N_k[i,:] = N_k_

        Q[i] = np.abs( N_k_ - np.abs(a_)**2 ).sum() / fo.n 

        u.repeat_print(('%i hrs, %i mins, %i s remaining.' %u.remaining(i + 1, fo.frames, fo.time0)))

    fo.t = t 
    fo.psi = psi
    fo.a = a 
    fo.N_x = N_x 
    fo.N_k = N_k

    return t, N_k, N_x, M, eigs, aa, a, psi, Q





def main(name, tags = []):
    time0 = time.time()

    setFigObj(name)

    fo.time0 = time.time()
    # perform analysis
    print("begining analysis...")
    t_cl, _, _, _, _, _, a_cl, psi_cl, _ = analyze('1F')
    fo.a_cl = a_cl
    fo.psi_cl = psi_cl

    fo.time0 = time.time()
    t, N_k, N_x, M, eigs, aa, a, psi, Q = analyze()

    SaveStuff(t, N_x, N_k, M, eigs, aa, a, Q, psi, psi_cl)

    print('completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0))

if __name__ == "__main__":
    main(simName)
    u.ding()