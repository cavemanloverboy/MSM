import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import animation
import os
import re



class figObj(object):

	def __init__(self):
		self.axXP = None # axis for x axis projection
		self.imXP = None

		self.axYP = None
		self.imYP = None

		self.axZP = None
		self.imZP = None

		self.axXS = None # axis for x axis slice
		self.imXS = None 

		self.axYS = None
		self.imYS = None

		self.axZS = None
		self.imZS = None

		self.time_text = None
		self.Tfinal = None

		self.N_images = None
		self.fileNames = None

		self.meta = None
fo = figObj()

# gets the names of files in the given directory organized by time stamp (ascending)
def getNames(name):
	files = [name + "/" + file for file in os.listdir(name) if (file.lower().endswith('.npy'))]
	files.sort(key=os.path.getmtime)
	return files


def getMetaKno(name):
	f = open(name + "/" + name + "Meta.txt")
	
	metaParams = {"C":None, "hbar_":None, "mpart":None, "L":None,
				"N":None, "Mtot":None, "dt":None, "Tfinal":None}

	for line in f.readlines():
		for key_ in metaParams.keys():
			if key_ + ":" in line:
				print(line)
				metaParams[key_] = re.findall(r"[-+]?\d*\.\d+|\d+", line)[0]

	return metaParams



def animate(i):
	T = float(i*float(fo.Tfinal))/1
	T /=(int(fo.N_images)-1)
	fo.time_text.set_text('$t=%.1f$' % T)

	psi = np.load(fo.fileNames[i])
	rho = float(fo.meta["Mtot"])*np.abs(psi)**2

	ind_ = int(fo.meta["N"])//2

	fo.imXP.set_array(rho.sum(axis=0))
	fo.imYP.set_array(rho.sum(axis=1))
	fo.imZP.set_array(rho.sum(axis=2))

	fo.imXS.set_array(rho[ind_,:,:])
	fo.imYS.set_array(rho[:,ind_,:])
	fo.imZS.set_array(rho[:,:,ind_])

	if np.min(rho) != np.max(rho):
	    fo.imXP.set_clim(vmin=np.min(rho), vmax=np.max(rho))
	    fo.imYP.set_clim(vmin=np.min(rho), vmax=np.max(rho))
	    fo.imZP.set_clim(vmin=np.min(rho), vmax=np.max(rho))

	    fo.imXS.set_clim(vmin=np.min(rho), vmax=np.max(rho))
	    fo.imYS.set_clim(vmin=np.min(rho), vmax=np.max(rho))
	    fo.imZS.set_clim(vmin=np.min(rho), vmax=np.max(rho))

	return fo.imXP, fo.imYP, fo.imZP, fo.imXS, fo.imYS, fo.imZS


def makeFig(meta, fo):
	fig, axs = plt.subplots(2,3)
	fo.axXP, fo.axYP, fo.axZP = axs[0]
	fo.axXS, fo.axYS, fo.axZS = axs[1]

	N = int(meta["N"])
	fo.Tfinal = float(meta["Tfinal"])

	fo.imXP = fo.axXP.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')
	fo.imYP = fo.axYP.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')
	fo.imZP = fo.axZP.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')

	fo.imXS = fo.axXS.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')
	fo.imYS = fo.axYS.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')
	fo.imZS = fo.axZS.imshow(np.zeros((N,N)), interpolation = 'none', aspect = 'auto',cmap = 'viridis')

	fo.axXP.set_title(r'$\rho$')
	fo.axYP.set_title(r'$\rho$')
	fo.axZP.set_title(r'$\rho$')

	fo.axXP.set_ylabel("projection")
	fo.axXS.set_ylabel("slice")

	fo.axXS.set_xlabel('x-axis')
	fo.axYS.set_xlabel('y-axis')
	fo.axZS.set_xlabel('z-axis')

	fo.time_text = fo.axXP.text(.7,.9,'', ha='center', va='center', transform=fo.axXP.transAxes, bbox = {'facecolor': 'white', 'pad': 5})

	return fig



def main(name = "testSim", int_ = 1000):
	fileNames = getNames(name)
	meta = getMetaKno(name)
	
	fo.meta = meta

	fig = makeFig(meta, fo)

	fo.N_images = len(fileNames)
	fo.fileNames = fileNames

	ani = animation.FuncAnimation(fig, animate,  frames=len(fileNames), interval=int_, blit=True)

	# for writing the video
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
	ani.save(name + ".mp4", writer=writer)	



if __name__ == "__main__":
	main()


