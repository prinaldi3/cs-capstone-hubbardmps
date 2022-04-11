import numpy as np
from matplotlib import pyplot as plt

nsteps = 2000
uot = 0.  # u over t0 ratio
maxerr = 1e-20

times = np.load("./Data/Tenpy/Basic/times-nsteps{}.npy".format(nsteps))
currents = np.load("./Data/Tenpy/Basic/currents-nsteps{}-U{}-err{}.npy".format(nsteps, uot, maxerr))
energies = np.load("./Data/Tenpy/Basic/energies-nsteps{}-U{}-err{}.npy".format(nsteps, uot, maxerr))

etimes = np.load("./Data/Exact/times-nsteps{}.npy".format(nsteps))
ecurrents = np.load("./Data/Exact/current-U{}-nsites10-nsteps{}.npy".format(uot, nsteps))
eenergies = np.load("./Data/Exact/energy-U{}-nsites10-nsteps{}.npy".format(uot, nsteps))

plt.plot(times, currents, label="MPS")
plt.plot(etimes, ecurrents, ls="dashed", label="Exact")
plt.legend()
plt.show()
