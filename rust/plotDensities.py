import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
import glob
import scipy.fftpack as sp
from pqdm.processes import pqdm
from time import time
import toml 

sim = "sim_data/bose-star-512-stream00001/psi*"
sim = "sim_data/gaussian-overdensity-512-mft/psi*"
# sim = "sim_data/spherical-tophat/psi*"
drops = np.sort(glob.glob(sim))
fps = 30


def radial_profile(drop):
	path_to_toml = "./examples/gaussian-overdensity-mft.toml"
	toml_file = open(path_to_toml, "r")
	toml_contents = toml_file.read()

	toml_dict = toml.loads(toml_contents)
	#print(toml_dict)

	#print(f"axis_length is {toml_dict['axis_length']}")
	#print(toml_dict['ics'])
	L = toml_dict['axis_length']
	N = toml_dict['size']
	hbar_ = toml_dict['hbar_']
	Mtot = toml_dict['total_mass']
	dx = L/N
	x = dx*(.5+np.arange(-N//2, N//2))
	u = hbar_*2*np.pi*sp.fftfreq(N, d = dx)

	psi = np.load(drop)

	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	rho = psi['real']**2 + psi['imag']**2)
	rho /= rho.sum()
	rho *= Mtot 

	R = np.linspace(dx, L/4, 32)

	grid = np.arange(rho.shape[0])*dx
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = L/2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())
	
	rho = [M_r[0] / (4.*np.pi) / (dx/2.)**2 / dx]
	for i in range(1,len(M_r)):
		r_diff = (R[i] + R[i-1])/2.
		M_diff = M_r[1] - M[i-1]
		rho.append(M_diff / (4.*np.pi) / (r_diff)**2 / dx)
	
	plt.figure(figsize=(24,8))
	plt.subplot(131)
	plt.xlabel(r"$x, \, [kpc]$")
	plt.ylabel(r"$y, \, [kpc]$")
	plt.title("Projected Spatial Density")

	plt.imshow(rho.mean(0), interpolation = 'none', 
        extent=[-L/2., L/2., -L/2,L/2], 
        aspect = 'auto', origin = 'lower',cmap = 'viridis')

	plt.subplot(132)
	plt.imshow(psi_k_sq.mean(0), interpolation = 'none', 
        extent=[np.min(u),np.max(u), np.min(u),np.max(u)], 
        aspect = 'auto', origin = 'lower',cmap = 'viridis')
	plt.xlabel(r"$u_x, \, [kpc / Myr]$")
	plt.ylabel(r"$u_y, \, [kpc / Myr]$")
	plt.title("Projected Momentum Density")


def end_and_final_radial_profile():

	drop0 = drops[0]
	psi = np.load(drop0)
	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	rho = psi['real']**2 + psi['imag']**2

if __name__ == "__main__":
	start = time()

	with imageio.get_writer('profile.mp4', fps=fps) as writer:

		images = pqdm(drops, radial_profile, n_jobs=25)

		for image in images:
			writer.append_data(imageio.imread(image))

	print(f"Finished visualization in {time()-start} seconds.")

	end_and_final_radial_profile()