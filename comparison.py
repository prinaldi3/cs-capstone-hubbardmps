import matplotlib.pyplot as plt
import numpy as np

nsteps = 2000
U = 10.0
nsites = 10

# format files containing the times at which states are evaluateed
mps_time_file = "./Data/EvolutionTesting/mps-times-nsteps{}.txt".format(nsteps)
exact_time_file = "./Data/EvolutionTesting/exact-times-nsteps{}.npy".format(nsteps)

# format files containing current density expectations
mps_file = "./Data/EvolutionTesting/mps-U{}-nsites{}-nsteps{}.txt".format(U, nsites, nsteps)
exact_file = "./Data/EvolutionTesting/exact-U{}-nsites{}-nsteps{}.npy".format(U, nsites, nsteps)

# load mps files
mps_data = []
with open(mps_file) as f:
    mps_data = f.readlines()

mps_times = []
with open(mps_time_file) as f:
    mps_times = f.readlines()

# first line gives number of steps, so it isn't really necessary
mps_data = mps_data[1:]
mps_times = mps_times[1:]

# load exact files
exact_data = np.load(exact_file)
exact_times = np.load(exact_time_file)

#clean the mps data so that it's a list of floats
for i in range(len(mps_data)):
    space_index = mps_data[i].find(" ")

    mps_data[i] = mps_data[i][:space_index]

    #handle scientific notation if necessary
    if "e" in mps_data[i]:
        e_index = mps_data[i].find("e")
        float_part = mps_data[i][:e_index]
        power = float(mps_data[i][e_index+1:])
        ten_factor = 10**power
        data_point = float(float_part) * ten_factor
        mps_data[i] = data_point
    else:
        mps_data[i] = float(mps_data[i])

mps_data = np.array(mps_data)
mps_data /= mps_data.max()
exact_data /= exact_data.max()

mps_times = [float(x) for x in mps_times]

# for i in range(len(mps_times)):
#     print(i, "-", mps_times[i], "-", mps_data[i])

# plot the comparison
plt.title("Comparison betwen MPS and Exact")
plt.xlabel("Time")
plt.ylabel("Current Density Expectation")

plt.plot(mps_times, mps_data, label="MPS")
plt.plot(exact_times, exact_data, label="Exact")
plt.legend()
plt.show()
