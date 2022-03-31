import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation
import os
import re
import utilsDark as u
import time

simName = "Test"

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

fo = figObj()

cuts = [[.7,.9], [.4,.6], [.1,.3]]


def animate(l):
	T = float(l*float(fo.Tfinal))/1
	T /=(int(fo.N_images)-1)
	T *= (int(fo.N_images)-1)/float(fo.drops)
	fo.time_text.set_text('$t=%.2f$' % T)

	ind_ = 0
	for i in range(len(fo.tags)):
		
		getCuts(fo,i)
			
		psi = np.load(fo.fileNames[i][l])
		rho = np.abs(psi)**2
		rhoS = [rho[fo.N[i]/2,:,:], rho[:,fo.N[i]/2,:], rho[:,:,fo.N[i]/2] ]

		for j in range(2): # projection vs slices
			for k in range(3): # dimensions 
				im_ = fo.ims[ind_]

				ar_ = None

				if j == 0:
					ar_ = performCut(rho.sum(axis = k), fo, k)
				else:
					ar_ = rhoS[k]

				im_.set_array(ar_)

				fo.currentMax[j] = np.max([np.max(ar_), fo.currentMax[j]])
				fo.currentMin[j] = np.min([np.min(ar_), fo.currentMin[j]])

				ind_ += 1

	ind_ = 0
	for i in range(len(fo.tags)):
		for j in range(2):
			for k in range(3):
				im_ = fo.ims[ind_]
				im_.set_clim(vmin = fo.currentMin[j], vmax = fo.currentMax[j])
				ind_ += 1

	u.repeat_print('%i hrs, %i mins, %i s remaining.' %u.remaining(l+1, fo.N_images, fo.start_))

	return fo.ims



def getCuts(fo,i):
	fo.xCut = [int(cuts[0][0]*fo.N[i]), int(cuts[0][1]*fo.N[i])]
	fo.yCut = [int(cuts[1][0]*fo.N[i]), int(cuts[1][1]*fo.N[i])]
	fo.zCut = [int(cuts[2][0]*fo.N[i]), int(cuts[2][1]*fo.N[i])]


def performCut(rho,fo,axis_):
	if axis_ == 0:
		fo.ex = np.array([fo.zCut[0],fo.zCut[1], fo.yCut[0],fo.yCut[1]])
		return rho[fo.yCut[0]:fo.yCut[1], fo.zCut[0]:fo.zCut[1]]
	if axis_ == 1:
		fo.ex = np.array([fo.zCut[0],fo.zCut[1], fo.xCut[0],fo.xCut[1]])
		return rho[fo.xCut[0]:fo.xCut[1], fo.zCut[0]:fo.zCut[1]]
	if axis_ == 2:
		fo.ex = np.array([fo.yCut[0],fo.yCut[1], fo.xCut[0],fo.xCut[1]])
		return rho[fo.xCut[0]:fo.xCut[1], fo.yCut[0]:fo.yCut[1]]	


def makeFig(fo):
	fig, axs = plt.subplots(len(fo.tags)*2,3, figsize = (30,20*len(fo.tags)))
	fo.axs = axs
	fo.ims = []

	N = fo.N

	titles = ["density projection, normal axis = x", "density slice, noraml axis = x"]
	dims = ["zy","zx","yx"]

	for i in range(len(fo.tags)): # simulation results
		
		getCuts(fo,i)

		for k in range(2): # projections vs slices
			
			for j in range(3): # dimensions
				i_ = 2*i + k
				ax_ = fo.axs[i_][j]

				if k == 0: # projections
					rho = np.zeros((N[i],N[i]))
					rho = performCut(rho, fo, j)
					fo.ex = float(fo.N[i])
				else: # slices
					fo.ex = (0,1,0,1)
					rho = np.zeros((N[i],N[i]))

				im_ = ax_.imshow(rho, interpolation = 'none', extent=fo.ex, aspect = 'auto',cmap = 'viridis', origin = 'lower')

				ax_.set_title(titles[k] + str(j+1))

				ax_.set_xlabel(dims[j][0] + "[L]")
				ax_.set_ylabel(dims[j][1] + "[L]")

				fo.ims.append(im_)

	fo.time_text = fo.axs[0,0].text(.7,.9,'', ha='center', va='center', transform=fo.axs[0,0].transAxes, bbox = {'facecolor': 'white', 'pad': 5})

	plt.tight_layout()
	return fig




def main(name, int_ = 1000):
	time0 = time.time()
	u.orientPlot()

	u.setFigObj(name, fo)
	u.setFileNames(fo, name)

	fig = makeFig(fo)

	fo.start_ = time.time()
	ani = animation.FuncAnimation(fig, animate, frames=fo.N_images, interval=int_)

	# for writing the video
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=8192, codec = "libx264")
	ani.save("../" + name + "_density_movie.mp4", writer=writer)
	print "output file: " + "../" + name + "_density_movie.mp4"

if __name__ == "__main__":
	main(simName)
