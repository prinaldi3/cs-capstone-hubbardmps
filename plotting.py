import numpy as np
from matplotlib import pyplot as plt
from tools import absolute_error, relative_error

nsteps = 2000
uot = 0.  # u over t0 ratio
maxerr = 1e-16
nsites = 10
pbc = False

mpsdir = "./Data/Tenpy/Basic/"
mpsparams = "-nsteps{}-U{}-err{}".format(nsteps, uot, maxerr)
# if pbc:
#     mpsparams += "-pbc"
# else:
#     mpsparams += "-obc"

exactdir = "./Data/Exact/"
exactparams = "-U{}-nsites{}-nsteps{}".format(uot, nsites, nsteps)
# if pbc:
#     exactparams += "-pbc"
# else:
#     exactparams += "-obc"

# times = np.load("./Data/Tenpy/Basic/times-nsteps{}.npy".format(nsteps))
# currents = np.load("./Data/Tenpy/Basic/currents-nsteps{}-U{}-err{}.npy".format(nsteps, uot, maxerr))
# energies = np.load("./Data/Tenpy/Basic/energies-nsteps{}-U{}-err{}.npy".format(nsteps, uot, maxerr))
times = np.load("./Data/Tenpy/Test/times.npy")
currents = np.load("./Data/Tenpy/Test/currents.npy")
energies = np.load("./Data/Tenpy/Test/energies.npy")


etimes = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
ecurrents = np.load("./Data/Exact/current-U{}-nsites{}-nsteps{}.npy".format(uot, nsites, nsteps))
eenergies = np.load("./Data/Exact/energy-U{}-nsites{}-nsteps{}.npy".format(uot, nsites, nsteps))

r_error = relative_error(ecurrents, currents)
print("relative_error:", r_error)

print(len(np.where((times - etimes) < 1e-5)[0]))

plt.plot(times, currents, label="MPS")
plt.plot(etimes, ecurrents, ls="dashed", label="Exact")
plt.legend()
plt.show()
