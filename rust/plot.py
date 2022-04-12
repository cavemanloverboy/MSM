import numpy as np
import matplotlib.pyplot as plt
import io
import imageio


ngrid = 128
ndumps = 200

with imageio.get_writer('coldgauss.mp4', fps=24) as writer:
	for i in range(1,ndumps):
		print(i)
		file = f"tests/data/cold-gauss_psi_{i}"
		psi = np.load(file)
		psi_sq = psi['real']**2 + psi['imag']**2
		psi_k = np.fft.fftshift(np.fft.fftn(psi['real'] + 1j*psi['imag']))
		psi_k_sq = np.abs(psi_k)**2



		plt.figure(1, figsize=(16,8))
		plt.subplot(121)
		plt.xlabel("x (grid cell)")
		plt.ylabel("y (grid cell)")
		plt.title("Projected Spatial Density")

		# this somewhat centers seed=0, N=128
		#plt.imshow(np.roll(np.roll(psi_sq.mean(0), -32, 1), -64,0))

		# this somehwat centers seed=0, N=256
		plt.imshow(np.roll(np.roll(psi_sq.mean(0), 0, 1), -128,0))

		# this somehwat centers seed=0, N=512, f64
		plt.imshow(np.roll(np.roll(psi_sq.mean(0), -328, 1), -100,0))

		plt.subplot(122)
		plt.imshow(psi_k_sq.mean(0))
		plt.xlabel("kx (grid cell)")
		plt.ylabel("ky (grid cell)")
		plt.title("Projected Momentum Density")

		buf = io.BytesIO()
		plt.savefig(buf)
		buf.seek(0)

		writer.append_data(imageio.imread(buf))
		plt.clf()
