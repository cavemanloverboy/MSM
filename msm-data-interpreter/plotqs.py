import numpy as np
import matplotlib.pyplot as plt

sim_name = "spherical-tophat-512"

N = 512 ** 3
x = np.load(f"../rust/sim_data/{sim_name}-combined/Qx_real")[:,0,0,0]
#k = np.load(f"../rust/sim_data/{sim_name}-combined/Qk_real")[:,0,0,0] / N

print(x)
#print(k)

#print(x-k)

plt.figure()
#plt.semilogy(x-x[0], linestyle="-", label="Qx")
plt.semilogy(x, linestyle="-", label="Qx")
plt.semilogy(np.ones_like(x)*0.5, linestyle="--")
#plt.semilogy(k-k[0], linestyle="--", label="Qk")
plt.xlabel("Dump")
plt.ylabel("Q")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Q")

plt.clf()
