import numpy as np
import matplotlib
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo',
# 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
# 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
print(plt.get_backend())

Nerrors = []
with open("./Data/Tenpy/GroundState/4Nto100N.log", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.find("INFO:tenpy.algorithms.dmrg:DMRG finished") != -1:
        x = lines[i - 3].split()
        Nerrors.append(x[3])

Uerrors = []
with open("./Data/Tenpy/GroundState/U_range.log", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.find("INFO:tenpy.algorithms.dmrg:DMRG finished") != -1:
        x = lines[i - 3].split()
        Uerrors.append(x[3])

Ns = list(range(4,101,4))
Ntimes = np.load("./Data/Tenpy/GroundState/times-4Nto100N.npy")
Us = [1/8, 1/4, 1/2, 1, 2, 4, 8]
Utimes = np.load("./Data/Tenpy/GroundState/times-U_range.npy")
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(Ns, Nerrors)
axs[0,1].plot(Ns, Ntimes)
axs[1,0].semilogx(Us, Uerrors, base=2)
axs[1,1].semilogx(Us, Utimes, base=2)
plt.show()
