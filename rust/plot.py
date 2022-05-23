import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
import glob
from pqdm.processes import pqdm
from time import time

sim = "sim_data/bose-star-512-stream00001/psi*"
sim = "sim_data/gaussian-overdensity-512-mft/psi*"
# sim = "sim_data/spherical-tophat/psi*"
drops = np.sort(glob.glob(sim))

fps = 30

def write_image(drop):

	psi = np.load(drop)
	psi_sq = psi['real']**2 + psi['imag']**2
	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	plt.figure(figsize=(16,8))
	plt.subplot(121)
	plt.xlabel("x (grid cell)")
	plt.ylabel("y (grid cell)")
	plt.title("Projected Spatial Density")

	# this somewhat centers seed=0, N=128
	#plt.imshow(np.roll(np.roll(psi_sq.mean(0), -32, 1), -64,0))

	# this somehwat centers seed=0, N=256
	#plt.imshow(np.roll(np.roll(psi_sq.mean(0), 0, 1), -128,0))

	# this somehwat centers seed=0, N=512, f64
	# plt.imshow(np.log10(np.sqrt(np.roll(np.roll(psi_sq.mean(0), -328, 1), -100,0))))
	plt.imshow(psi_sq.mean(0))

	plt.subplot(122)
	plt.imshow(psi_k_sq.mean(0))
	plt.xlabel("kx (grid cell)")
	plt.ylabel("ky (grid cell)")
	plt.title("Projected Momentum Density")

	buf = io.BytesIO()
	plt.savefig(buf)
	buf.seek(0)

	return buf

# start = time()

# with imageio.get_writer('bose-star.mp4', fps=fps) as writer:

# 	images = pqdm(drops, write_image, n_jobs=25)

# 	for image in images:
# 		writer.append_data(imageio.imread(image))

# print(f"Finished visualization in {time()-start} seconds.")

def radial_profile(drop):
    	
	psi = np.load(drop)
	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	rho = psi['real']**2 + psi['imag']**2

	R = np.linspace(1, rho.shape[0]//2, 16)

	grid = np.arange(rho.shape[0])
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = rho.shape[0]//2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())

	plt.figure(figsize=(24,8))
	plt.subplot(131)
	plt.xlabel("x (grid cell)")
	plt.ylabel("y (grid cell)")
	plt.title("Projected Spatial Density")

	# this somewhat centers seed=0, N=128
	#plt.imshow(np.roll(np.roll(psi_sq.mean(0), -32, 1), -64,0))

	# this somehwat centers seed=0, N=256
	#plt.imshow(np.roll(np.roll(psi_sq.mean(0), 0, 1), -128,0))

	# this somehwat centers seed=0, N=512, f64
	# plt.imshow(np.log10(np.sqrt(np.roll(np.roll(psi_sq.mean(0), -328, 1), -100,0))))
	plt.imshow(rho.mean(0))

	plt.subplot(132)
	plt.imshow(psi_k_sq.mean(0))
	plt.xlabel("kx (grid cell)")
	plt.ylabel("ky (grid cell)")
	plt.title("Projected Momentum Density")

	plt.subplot(133)
	plt.cla()
	plt.plot(R, M_r)
	plt.xlabel("pixels")
	plt.ylabel("mass enclosed within pixels")
	plt.title("Enclosed mass")

	buf = io.BytesIO()
	plt.savefig(buf)
	buf.seek(0)

	return buf
    	
start = time()

with imageio.get_writer('profile.mp4', fps=fps) as writer:

	images = pqdm(drops, radial_profile, n_jobs=25)
	# images = []
	# for drop in drops:
	# 		images.append(radial_profile(drop))

	for image in images:
		writer.append_data(imageio.imread(image))

print(f"Finished visualization in {time()-start} seconds.")


def end_and_final_radial_profile():
    	
	drop0 = drops[0]
	psi = np.load(drop0)
	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	rho = psi['real']**2 + psi['imag']**2

	R = np.arange(1, rho.shape[0]//2)

	grid = np.arange(rho.shape[0])
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = rho.shape[0]//2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())

	plt.figure(figsize=(12,8))
	plt.plot(R, M_r, label="initial")

	drop_final = drops[-1]
	psi = np.load(drop_final)
	psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
	psi_k_sq = np.abs(psi_k)**2

	rho = psi['real']**2 + psi['imag']**2

	R = np.arange(1, rho.shape[0]//2)

	grid = np.arange(rho.shape[0])
	xx, yy, zz = np.meshgrid(grid, grid, grid)
	center = rho.shape[0]//2
	diff = np.sqrt( (xx-center)**2 + (yy-center)**2 + (zz-center)**2)

	M_r = []
	for r in R:
		M_r.append(rho[diff < r].sum())

	plt.plot(R, M_r, label="final")

	plt.xlabel("pixels")
	plt.ylabel("mass enclosed within pixels")
	plt.title("Enclosed mass")
	plt.legend()

	plt.savefig("profile.png", dpi=200)
end_and_final_radial_profile()