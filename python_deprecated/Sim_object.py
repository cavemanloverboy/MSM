import numpy as np 
import time
import datetime
import os 
import utilsDark as u
from decimal import Decimal

class Sim(object):

	def __init__(self, ofile = "test", Tfinal = 1., drops = 100):

		self.time0 = None

		self.solvers = None

		self.ofile = ofile

		self.Tfinal = Tfinal

		self.drops = drops


	def setSolvers(self, solvers):
		self.solvers = solvers

	def PrepareSim(self):
		try:
			os.mkdir("../" + self.ofile + "/")
		except OSError:
			pass
		for solver_ in self.solvers:
			solver_.readyDir(self.ofile)

	# makes meta file for each solver with appropriate tag and
	# makes sim meta file that contains tags
	def MakeStandardMetaFiles(self):
		tags = ""
		for solver_ in self.solvers:
			self.MakeMetaFileWithTag(solver_.tag, N = solver_.N, M_BH_frac = solver_.M_BH_frac,
				L = solver_.L, hbar_ = solver_.hbar_, chi = solver_.chi, dtau = solver_.dtau,
				phi_max = solver_.phi_max, phi_min = solver_.phi_min, dtype_ = solver_.dtype_)
			tags += "," + solver_.tag
		self.MakeMetaFileWithTag("", tags = tags[1:], Tfinal = self.Tfinal, drops = self.drops)


	def MakeMetaFileWithTag(self, tag, **kwargs):
		try:
			os.mkdir("../" + self.ofile + "/")
		except OSError:
			pass
		f = open("../" + self.ofile + "/" + tag + "Meta.txt", 'w+')
		f.write("sim start: " + str(datetime.datetime.now()) + '\n\n')

		for key, value in kwargs.items():
			f.write(str(key) + ": " + str(value) + '\n')
		f.write('\n')

		f.close()

	def run(self):
		for i in range(self.drops):
			start_ = time.time()
			Tf = self.Tfinal*(i+1.)/float(self.drops)
			dtau = None
			for solver_ in self.solvers:
				solver_.dropData(i, self.ofile)
				failed = solver_.update(Tf)
				if dtau is None:
					dtau = solver_.dtau
				else:
					dtau = np.min([dtau, solver_.dtau])
				if failed:
					print "solver encountered event type: \"" + failed + "\" by T =", Tf
			str_ = '%i percent compelete. dtau = %.2E. estimated ' %(100.*(i+1.)/self.drops, Decimal(str(dtau)))
			str_ += ('%i hrs, %i mins, %i s remaining.' %u.remaining(1, self.drops - 1 - i, start_))
			u.repeat_print(str_)

	def endSim(self):
		for solver_ in self.solvers:
				solver_.dropData(self.drops, self.ofile)
		print '\nsim completed in %i hrs, %i mins, %i s' %u.hms(time.time()-self.time0)
		print "output directory: ", self.ofile
