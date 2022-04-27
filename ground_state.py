
"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time

import logging
logging.basicConfig(level=logging.INFO, filename="./Data/Tenpy/GroundState/4Nto100N.log")

# energy parameters, in units eV
it = .52
##########################
"""IMPORTANT PARAMETERS"""
##########################
phi_func = phi_tl
maxerr = 1e-12  # used for DMRG
pbc = False
N = 10
iU = 1 * it
pbc = False  # periodic boundary conditions
nsteps = 2000

"""We will hold these parameters constant"""
# lattice spacing, in angstroms
ia = 4
# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

Ns = list(range(4, 101, 4))
print(Ns)
times = []

for N in Ns:
    p = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

    # get the start time
    start_time = time.time()
    model = FHHamiltonian(p, 0)
    sites = model.lat.mps_sites()
    state = ["up", "down"] * (N // 2)
    psi0_i = MPS.from_product_state(sites, state)

    ti = time.time()
    # the max bond dimension
    chi_list = {0:20, 1:40, 2:100, 4:200, 6:400, 8:800}
    dmrg_dict = {"chi_list":chi_list, "max_E_err":maxerr, "max_sweeps":10, "mixer":True, "combine":False}
    dmrg_params = Config(dmrg_dict, "DMRG-maxerr{}".format(maxerr))
    dmrg = DMRG(psi0_i, model, dmrg_params)
    E, psi0 = dmrg.run()
    tot_time = time.time() - ti
    print("Total time: {:.4f}".format(tot_time))
    times.append(tot_time)

np.save("./Data/Tenpy/GroundState/times-4Nto100N.npy", times)

plt.plot(Ns, times)
plt.xlabel("System Size")
plt.ylabel("Time (seconds)")
