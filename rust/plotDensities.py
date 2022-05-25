import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
import glob
import scipy.fftpack as sp
from pqdm.processes import pqdm
from time import time
import toml
import gc

sim_name = "gaussian-overdensity-512-stream00001"
sim = f"sim_data/{sim_name}/psi_?????_real"
# sim = "sim_data/spherical-tophat/psi*"
drops = np.sort(glob.glob(sim))

fps = 20


shell_volume = lambda R, r: 4 * np.pi / 3 * (R**3 - r**3)

path_to_toml = f"./examples/go-streams/{sim_name}.toml"
toml_file = open(path_to_toml, "r")
toml_contents = toml_file.read()

toml_dict = toml.loads(toml_contents)
output_name = f"{toml_dict['sim_name']}.mp4"

def radial_profile(drop):

	#print(toml_dict)

	#print(f"axis_length is {toml_dict['axis_length']}")
	#print(toml_dict['ics'])
	L = float(toml_dict['axis_length'])
	N = int(toml_dict['size'])
	hbar_ = float(toml_dict['hbar_'])
	Mtot = float(toml_dict['total_mass'])
	dx = L/N
	x = dx*(.5+np.arange(-N//2, N//2))
	u = hbar_*2*np.pi*sp.fftfreq(N, d = dx)

	# this is psi, but naming rho to reuse memory
	rho = np.load(f"{drop}") + 1j*np.load(f"{drop[:-5]}_imag")

	# getting psi_k_sq out of psi, presently named rho
	psi_k_sq = np.abs(np.fft.fftshift(np.fft.fftn(rho)))**2

	# after this it is rho, psi is discarded
	rho = np.abs(rho)**2
	rho /= rho.sum()
	rho *= Mtot 

	R = np.unique(np.logspace(np.log10(1.01), np.log10(N//2)+0.01, 32).astype(int)) * dx

	grid = (np.arange(rho.shape[0]) + 1/2)*dx
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = L/2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())
	

	rho_r = [M_r[0] / shell_volume(dx,0)]
	for i in range(1,len(M_r)):
		M_diff = M_r[i] - M_r[i-1]
		rho_r.append(M_diff / shell_volume(R[i], R[i-1]))

	plt.figure(figsize=(24,8))
	plt.subplot(131)
	plt.xlabel(r"$x\ [kpc]$")
	plt.ylabel(r"$y\ [kpc]$")
	plt.title("Projected Spatial Density")
	plt.imshow(rho.mean(0), interpolation = 'none', 
        extent=[-L/2., L/2., -L/2,L/2], 
        aspect = 'auto', origin = 'lower', cmap = 'viridis')

	plt.subplot(132)
	plt.imshow(psi_k_sq.mean(0), interpolation = 'none', 
        extent=[np.min(u),np.max(u), np.min(u),np.max(u)], 
        aspect = 'auto', origin = 'lower',cmap = 'viridis')

	plt.xlabel(r"$u_x \, [kpc / Myr]$")
	plt.ylabel(r"$u_y \, [kpc / Myr]$")
	plt.title("Projected Momentum Density")

	plt.subplot(133)
	plt.loglog(R, rho_r)
	plt.xlabel(r"$R\ [kpc]$")
	plt.ylabel(r"$\rho(R), [M_\odot\ kpc^{-3}]$")
	plt.title("Radial Density Profile")
	plt.ylim(1e4, 1e12)

	buf = io.BytesIO()
	plt.savefig(buf)
	buf.seek(0)

	gc.collect()

	return buf

def initial_and_final_radial_profile():
    	
	Mtot = float(toml_dict['total_mass'])
	L = float(toml_dict['axis_length'])
	N = int(toml_dict['size'])

	drop0 = drops[0]
	psi = rho = np.load(f"{drop0}") + 1j*np.load(f"{drop0[:-5]}_imag")
	
	rho = np.abs(psi)**2
	rho /= rho.sum()
	rho *= Mtot 

	dx = L/N

	R = np.linspace(dx, L/4, 32)

	grid = np.arange(rho.shape[0])*dx
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = L/2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())

	plt.figure(figsize=(12,8))
	plt.loglog(R, M_r, label="initial")

	drop_final = drops[-1]
	psi = rho = np.load(f"{drop_final}") + 1j*np.load(f"{drop_final[:-5]}_imag")

	rho = np.abs(psi)**2
	rho /= rho.sum()
	rho *= Mtot 

	R = np.linspace(dx, L/4, 32)

	grid = np.arange(rho.shape[0])*dx

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())
	

	plt.loglog(R, M_r, label="final")

	plt.xlabel("Radius")
	plt.ylabel("M(<R)")
	plt.title("Enclosed mass")
	plt.legend()

	plt.savefig("profile.png", dpi=200)

if __name__ == "__main__":
	start = time()

	with imageio.get_writer(output_name, fps=fps) as writer:

		images = pqdm(drops, radial_profile, n_jobs=20)
		# images = []
		# for drop in drops:
		# 	images.append(radial_profile(drop))

		for image in images:
			writer.append_data(imageio.imread(image))

	print(f"Finished visualization in {time()-start} seconds.")

	initial_and_final_radial_profile()