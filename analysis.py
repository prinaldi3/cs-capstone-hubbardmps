import numpy as np
from matplotlib import pyplot as plt

io = open("./Data/Tenpy/expectations-U0.0.txt", "r")
lines = io.readlines()
io.close()

times = []
energies = []
currents = []
for line in lines:
    x = [y.strip("()\n") for y in line.split(", ")]
    times.append(float(x[0]))
    energies.append(float(x[1]))
    currents.append(complex(x[2]).real)

eenergies = np.load("./Data/Exact/energy-U0.0-nsites10-nsteps2000.npy")
etimes = np.load("./Data/Exact/times-nsteps2000.npy")

plt.plot(times, currents, label="MPS")
# plt.plot(etimes, eenergies, ls="dashed", label="Exact")
plt.legend()
plt.show()
