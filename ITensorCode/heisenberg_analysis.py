import numpy as np
from matplotlib import pyplot as plt

taus = [.1, .05, .01, .005, .001]
# taus = [.1, .05, .01, .005, .001, .0005, .0001]
tf = 5.0
nsteps = [int(tf / tau) for tau in taus]

alltimes = []
allenergies = []
for steps in nsteps:
    x = []
    with open("./Data/HeisenbergTest/nsteps{}.txt".format(steps)) as f:
        x = f.readlines()
    times = []
    energies = []
    for line in x:
        temp = line.split(", ")
        times.append(float(temp[0]))
        energies.append(float(temp[1]))
    alltimes.append(times)
    allenergies.append(energies)

for i in range(len(taus)):
    plt.plot(alltimes[i], allenergies[i], label="{}".format(nsteps[i]))

plt.legend()
plt.show()
