from tenpy.tools.params import Config
# from tenpy.models.hubbard import FermiHubbardChain as FH
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine as DMRG
from tenpy.networks.mps import MPS
# from tenpy.algorithms.tebd import Engine as TEBD
from tebd import Engine as TEBD
from tenpy.algorithms.truncation import TruncationError
from tools import Parameters, phi_tl, FHHamiltonian, FHCurrent
import numpy as np

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

# model_dict = {"bc_MPS":"finite", "cons_N":"N", "cons_Sz":"Sz", "explicit_plus_hc":True,
# "L":p.nsites, "mu":0, "V":0, "U":p.u, "t":p.t0}
# model_params = Config(model_dict, "FH-U{}".format(p.u))

model = FHHamiltonian(0, p, phi_tl)
current = FHCurrent(0, p, phi_tl)
sites = model.lat.mps_sites()
state = ["up", "down"] * (N // 2)
psi0_i = MPS.from_product_state(sites, state)

maxerr = 1e-10
# the max bond dimension
chi_list = {0:20, 1:40, 2:100, 4:200, 6:400, 8:800}
dmrg_dict = {"chi_list":chi_list, "max_E_err":maxerr, "max_sweeps":10, "mixer":True, "combine":False}
dmrg_params = Config(dmrg_dict, "DMRG-maxerr{}".format(maxerr))
dmrg = DMRG(psi0_i, model, dmrg_params)
E, psi0 = dmrg.run()

print("Current:", current.H_MPO.expectation_value(psi0))

psi = psi0

ti = 0
tf = 2 * np.pi * cycles / p.field
nsteps = 2000
delta = (tf - ti) / nsteps

tebd_dict = {"dt":delta, "order":2, "start_time":ti, "start_trunc_err":TruncationError(eps=maxerr), "trunc_params":{"trunc_cut":maxerr}, "N_steps":nsteps}
tebd_params = Config(tebd_dict, "TEBD-trunc_err{}-nsteps{}".format(maxerr, nsteps))
tebd = TEBD(psi, model, p, phi_tl, tebd_params)
tebd.run()
