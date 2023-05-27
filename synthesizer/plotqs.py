import numpy as np
import matplotlib.pyplot as plt

sim_name = "gaussian-overdensity-512-lown"

N = 512 ** 3
x = np.load(f"../sim-data/{sim_name}-combined/Qx_real")[:,0,0,0]
#k = np.load(f"../rust/sim-data/{sim_name}-combined/Qk_real")[:,0,0,0] / N

print(x)
#print(k)

#print(x-k)

n = 3.6e8
sym_corr = N / 2 / n
q0 = x[0]
Q = x - q0 - sym_corr
plt.figure()
plt.semilogy(x-x[0], linestyle="-", label="Qx")
#plt.semilogy(x, linestyle="-", label="Qx")
plt.semilogy(np.ones_like(x)*0.5, linestyle="--")
#plt.semilogy(k-k[0], linestyle="--", label="Qk")
plt.xlabel("Dump")
plt.ylabel("Q")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Q")

plt.clf()
