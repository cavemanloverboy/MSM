import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation
import os
import re
import utilsDark as u
import time

simName = "Tkachev"

fo = u.figObj()


def makeFig(fo):
	fig, axs = plt.subplots((len(fo.tags)*2,3), figsize = (15,10*len(fo.tags)))
	fo.axs = axs
	fo.ims = []

	N = fo.N

	titles = ["density projection, normal axis = x", "density slice, noraml axis = x"]
	dims = ["yz","xz","xy"]

	for i in range(len(fo.tags)): # simulation results
		
		for k in range(2): # projections vs slices
			
			for j in range(3): # dimensions
				i_ = 2*i + k
				ax_ = fo.axs[i_][j]

				im_ = ax_.imshow(np.zeros((N,N)), interpolation = 'none', extent=[0, 1, 0, 1], aspect = 'auto',cmap = 'viridis')

				ax_.set_title(titles[k] + str(j+1))

				ax_.set_xlabel(dims[j][0] + "[L]")
				ax_.set_ylabel(dims[j][1] + "[L]")

				fo.ims.append(im_)

	return fig




def main(name):
	time0 = time.time()
	u.orientPlot()

	u.setFigObj(name, fo)
	u.setFileNames(fo, name)

	fig = makeFig(fo)


if __name__ == "__main__":
	main(simName)
