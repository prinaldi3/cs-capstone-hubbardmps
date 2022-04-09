from tenpy.tools.params import Config
# from tenpy.models.hubbard import FermiHubbardChain as FH
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
# from tenpy.algorithms.tebd import Engine as TEBD
from tebd import Engine as TEBD
from tenpy.algorithms.truncation import TruncationError
from tools import Parameters, phi_tl, FHHamiltonian, FHCurrent
import numpy as np

from multiprocessing import Pool

N = 10
# energy parameters, in units eV
it = .52
iU = 0.5 * it

# lattice spacing, in angstroms
ia = 4

# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

p = Parameters(N, iU, it, ia, cycles, iomega0, iF0)

"""Set the function for phi"""
phi_func = phi_tl
"""Cut off error"""
maxerr = 1e-15

data = []
for uot in [0, .125, .25, .5, 1, 2, 4, 8]:
    for maxerr in [1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20]:
        data.append((Parameters(N, uot * it, it, ia, cycles, iomega0, iF0), maxerr))

def runsim(p, maxerr):

    model = FHHamiltonian(0, p, phi_func)
    current = FHCurrent(0, p, phi_func)
    sites = model.lat.mps_sites()
    state = ["up", "down"] * (N // 2)
    psi0_i = MPS.from_product_state(sites, state)

    # the max bond dimension
    chi_list = {0:20, 1:40, 2:100, 4:200, 6:400, 8:800}
    dmrg_dict = {"chi_list":chi_list, "max_E_err":maxerr, "max_sweeps":10, "mixer":True, "combine":False, "verbose":0}
    dmrg_params = Config(dmrg_dict, "DMRG-maxerr{}".format(maxerr))
    dmrg = DMRG(psi0_i, model, dmrg_params)
    E, psi0 = dmrg.run()

    psi = psi0

    ti = 0
    tf = 2 * np.pi * cycles / p.field
    nsteps = 2000
    delta = (tf - ti) / nsteps

    tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"svd_min":maxerr, "chi_max":1000}, "N_steps":nsteps, "verbose":0}
    tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
    tebd = TEBD(psi, model, p, phi_tl, tebd_params)
    times, energies, currents = tebd.run()

    savedir = "./Data/Tenpy/Basic/"
    allps = "-nsteps{}".format(nsteps)
    ecps = "-U{}-err{}".format(p.u, maxerr)
    np.save(savedir + "times" + allps + ".npy", times)
    np.save(savedir + "energies" + allps + ecps + ".npy", energies)
    np.save(savedir + "currents" + allps + ecps + ".npy", currents)

pool = Pool()
pool.starmap(runsim, data)
