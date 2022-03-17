import matplotlib.pyplot as plt
import numpy as np

nsteps = 2000
U = 0.5
nsites = 10

method = "RK4"
# method = "TEBD"

# analyzing = "current"
analyzing = "energy"

# format file names for files containing the times at which states are evaluated
mps_time_file = "./Data/MPSTimes/times-nsteps{}.txt".format(nsteps)
exact_time_file = "./Data/Exact/times-nsteps{}.npy".format(nsteps)

# format file names for files containing expectations
mps_file = "./Data/{}/mps-{}-U{}-nsites{}-nsteps{}.txt".format(method, analyzing, U, nsites, nsteps)
exact_file = "./Data/Exact/{}-U{}-nsites{}-nsteps{}.npy".format(analyzing, U, nsites, nsteps)

# load mps files
mps_data = []
with open(mps_file) as f:
    mps_data = f.readlines()

mps_times = []
with open(mps_time_file) as f:
    mps_times = f.readlines()

# load exact files
exact_data = np.load(exact_file)
exact_times = np.load(exact_time_file)

# turn the strings into floats
mps_times = [float(x) for x in mps_times]
mps_data = [float(x) for x in mps_data]
mps_data = np.array(mps_data)

# optional: normalize data
# mps_data /= mps_data.max()
# exact_data /= exact_data.max()

# plot the comparison
plt.title("Comparison betwen MPS and Exact")
plt.xlabel("Time")
plt.ylabel("Current Density Expectation")

plt.plot(mps_times, mps_data, label="MPS")
plt.plot(exact_times, exact_data, label="Exact")
plt.legend()
plt.show()
