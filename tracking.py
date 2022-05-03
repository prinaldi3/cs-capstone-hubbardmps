
"""Tenpy packages"""
from tenpy.tools.params import Config
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
from tebd import Engine as TEBD
from tenpy.algorithms.truncation import TruncationError
from tools import *

import numpy as np
from matplotlib import pyplot as plt
import time

import logging
logging.basicConfig(level=logging.INFO)

# energy parameters, in units eV
it = .52
##########################
"""IMPORTANT PARAMETERS"""
##########################
phi_func = phi_tracking
maxerr = 1e-12 # used for DMRG
maxdim = 800 # maximum bond dimension, used for TEBD
pbc = False
N = 20
iU = 0 * it
pbc = False  # periodic boundary conditions
nsteps = 4000

"""We will hold these parameters constant"""
# lattice spacing, in angstroms
ia = 4
# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

p = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

#########################
"""TRACKING PARAMETERS"""
#########################
tuot = 1.  # tracking U/t_0
dir = "./Data/Tenpy/Basic/"
ps = "-nsteps{}-nsites{}-U{}-maxdim{}".format(nsteps, p.nsites, tuot, maxdim)
tracking_current = np.load(dir + "currents" + ps + ".npy")

# get the start time
start_time = time.time()
# it should be okay to create the first operators with phi_tl because at time 0 current should be 0
model = FHHamiltonian(p, 0)
current = FHCurrent(p, 0)
sites = model.lat.mps_sites()
state = ["up", "down"] * (N // 2)
psi0_i = MPS.from_product_state(sites, state)

# the max bond dimension
chi_list = {0:20, 1:40, 2:100, 4:200, 6:400, 8:maxdim}
dmrg_dict = {"chi_list":chi_list, "max_E_err":maxerr, "max_sweeps":10, "mixer":True, "combine":False}
dmrg_params = Config(dmrg_dict, "DMRG-maxerr{}".format(maxerr))
dmrg = DMRG(psi0_i, model, dmrg_params)
E, psi0 = dmrg.run()

psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
times, delta = np.linspace(ti, tf, num=nsteps, endpoint=True, retstep=True)
# we pass in nsteps - 1 because we would like to evauluate the system at
# nsteps time points, including the ground state calculations
tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":maxdim}, "N_steps":nsteps-1}
tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
tebd = TEBD(psi, model, p, phi_func, tebd_params)
times, energies, currents, phis = tebd.run(tracking_current)

tot_time = time.time() - start_time

print("Evolution complete, total time:", tot_time)

error = relative_error(tracking_current, currents)
print("Total error:", error)

savedir = "./Data/Tenpy/Tracking/"
allps = "-nsteps{}".format(nsteps)
ecps = "-nsites{}-sU{}-tU{}-maxdim{}".format(p.nsites, p.u, tuot, maxdim, maxerr)
np.save(savedir + "currents" + allps + ecps + ".npy", currents)
np.save(savedir + "phis" + allps + ecps + ".npy", phis)

plt.plot(times, currents)
plt.plot(times, tracking_current, label="Tracked Current", ls="dashed")
plt.legend()
plt.show()
