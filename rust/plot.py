import numpy as np
import matplotlib.pyplot as plt
import io
import imageio
import glob
from pqdm.processes import pqdm
from time import time

sim = "sim_data/bose-star-512/psi*"
drops = np.sort(glob.glob(sim))

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
	plt.imshow(np.log10(np.sqrt(np.roll(np.roll(psi_sq.mean(0), -328, 1), -100,0))))

	plt.subplot(122)
	plt.imshow(psi_k_sq.mean(0))
	plt.xlabel("kx (grid cell)")
	plt.ylabel("ky (grid cell)")
	plt.title("Projected Momentum Density")

	buf = io.BytesIO()
	plt.savefig(buf)
	buf.seek(0)

	return buf

    	
start = time()

with imageio.get_writer('bose-star.mp4', fps=24) as writer:

	images = pqdm(drops, write_image, n_jobs=20)

	for image in images:
		writer.append_data(imageio.imread(image))

print(f"Finished visualization in {time()-start} seconds.")
