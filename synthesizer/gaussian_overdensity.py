import numpy as np
import matplotlib.pyplot as plt

# Reuse one figure for all plots
plt.figure()

# Plot Q for a simulation
def plot_q(sim_name, Ncell, ntot, label):
    # Load raw Q value
    x = np.load(f"../sim-data/{sim_name}-combined/Qx_real")[:,0,0,0]

    # Offset and correct Q
    sym_corr = Ncell / 2 / ntot
    q0 = x[0]
    Q = x - q0 - sym_corr

    plt.semilogy(x-x[0], linestyle="-", label=label)
    plt.xlabel("Dump")
    plt.ylabel("Q")
    plt.grid()
    plt.legend()
    plt.tight_layout()



# Low N
sim_name = "gaussian-overdensity-512-lown"
N = 512 ** 3
n = 3.6e9
plot_q(sim_name, N, n, "low")

# High N
#sim_name = "gaussian-overdensity-512-mft"
#N = 512 ** 3
#n = 3.6e11
#plot_q(sim_name, N, n, "hi")

# Super Low N
sim_name = "gaussian-overdensity-512-slow"
N = 512 ** 3
n = 3.6e9
#plot_q(sim_name, N, n)

# High N 2
sim_name = "gaussian-overdensity-512-part2-highn"
N = 512 ** 3
n = 3.6e11
plot_q(sim_name, N, n, "highn2")

# High N 2
sim_name = "gaussian-overdensity-512-part2-lown"
N = 512 ** 3
n = 3.6e11
plot_q(sim_name, N, n, "lown2")

plt.savefig(f"{sim_name}")
plt.clf()



