import matplotlib.pyplot as plt
import numpy as np

mps_data = []
with open('sim_data_file.txt') as f:
    mps_data = f.readlines()

exact_data = list(np.load("exact_data_file.npy"))

mps_data = mps_data[1:]
exact_data = exact_data[1:]


#clean the mps data so that it's a list of floats
for i in range(len(mps_data)):
    space_index = mps_data[i].find(" ")

    mps_data[i] = mps_data[i][:space_index]

    #handle scientific notation if necessary
    if "e" in mps_data[i]:
        e_index = mps_data[i].find("e")
        float_part = mps_data[i][:e_index]
        power = mps_data[i][e_index+1:]
        ten_factor = 10**power
        data_point = float(float_part) * ten_factor
        mps_data[i] = data_point
    else:
        mps_data[i] = float(mps_data[i])

#plot the comparison
plt.title("Comparison betwen MPS and Exact")
plt.xlabel("Time")
plt.ylabel("Current Density Expectation")

plt.plot(mps_data, label="MPS")
plt.plot(exact_data, label="Exact")
plt.legend()