import numpy as np
from matplotlib import pyplot as plt
from tools import relative_error

nsteps = 2000
nsites = 10
uot = 1. # u over t0 ratio
maxdim = 1000
pbc = False

mpsdir = "./Data/Tenpy/Basic/"
mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, nsites, uot, maxdim)

exactdir = "./Data/Exact/"
exactparams = "-U{}-nsites{}-nsteps{}".format(uot, nsites, nsteps)

times = np.load(mpsdir + "times-nsteps{}.npy".format(nsteps))
# currents = np.load(mpsdir + "currents" + mpsparams + ".npy")
# energies = np.load(mpsdir + "energies" + mpsparams + ".npy")
#
#
# etimes = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
# ecurrents = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}-pbc.npy".format(uot, 10, nsteps))
# eenergies = np.load("./Data/Exact/energy-U{}-nsites{}-nsteps{}.npy".format(uot, 10, nsteps))
#
#
# plt.plot(times, currents, label="MPS")
# plt.plot(etimes, ecurrents, ls="dashed", label="Exact")
# plt.legend()
# plt.show()

times = []
times0 = []
size = []
for N in range(6, 19, 2):
    mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, uot, 100 * N)
    with open(mpsdir + "metadata" + mpsparams + ".txt") as f:
        size.append(N)
        times.append(float(f.readlines()[0]))
    mpsparams = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, N, 0., 100 * N)
    with open(mpsdir + "metadata" + mpsparams + ".txt") as f:
        times0.append(float(f.readlines()[0]))

plt.plot(size, times0, "o", label="$\\frac{U}{t_0} = 0$")
plt.plot(size, times, "o", label="$\\frac{U}{t_0} = 1$")
plt.legend()
plt.xlabel("System size")
plt.ylabel("Time (seconds)")
plt.show()

# params = ["-nsteps{}-nsites{}-U{}-maxdim{}-maxerr{}".format(nsteps, N, uot, 100 * N, maxerr) for N in range(12, 19, 2)]
# params.append("-nsteps{}-nsites{}-U{}-maxdim{}-maxerr{}".format(nsteps, 20, uot, 2400, 1e-12))
# data = [np.load(mpsdir + "currents" + param + ".npy") for param in params]
#
# for i, N in enumerate(range(12, 21, 2)):
#     plt.plot(times, data[i] / N, label="{}".format(N))
# plt.legend()
# plt.show()
