import utilsDark as u 
import SP3D_object as SP 
import Sim_object as Sim 
import time
import numpy as np
import cupy as cp
import di_3D_movie

ofile = "Test"

drops = 100

chi = 1e5
dtau = 1e-7

Tfinal = 1.

N = 256

dIs = [di_3D_movie]


def main():
	time0 = time.time()
	print "begining sim: ", ofile

	s = Sim.Sim(ofile = ofile, Tfinal = Tfinal, drops = drops)

	test = np.ones((N,N,N))*1. +0j
	gB_expected = test.nbytes/1e9

	sp = SP.SP3D_cp(N = N, chi = chi, dtau = dtau, parrallel = False)
	solvers = [sp]
	s.setSolvers(solvers)

	s.PrepareSim()
	s.MakeStandardMetaFiles()
	
	memUsage = 0
	for i in range(len(s.solvers)):
		s.solvers[i].Tkachev()
		memUsage += s.solvers[i].psi.nbytes*drops

	print 'initialization completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
	print "begining sim, expected memory usage: ", memUsage/1e9, " Gb"

	s.time0 = time.time()
	s.run()
	s.endSim()

	time0 = time.time()
	print "begining data interpretation"

	for i in range(len(dIs)):
		dIs[i].main(ofile)
	print 'analysis completed in %i hrs, %i mins, %i s' %u.hms(time.time()-time0)
	u.ding()

if __name__ == "__main__":
	main()