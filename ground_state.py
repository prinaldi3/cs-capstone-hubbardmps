"""
In this file we would like to create the ground state of the half-filled
1D Hubbard model with onsite coupling and compare expectation values
to those generated from the exact state in QuSpin.
"""
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
from time import time
from tools import Parameters
from tenpy.simulations.ground_state_search import GroundStateSearch
from tenpy.models.hubbard import FermiHubbardModel


"""Model parameters"""
nsites = 20
t0 = 0.52
u = 1 * t0
bc = "periodic"

params = Parameters(nsites, u, t0, bc)
'''
"""Exact calculation of ground state using QuSpin"""
ti = time()
basis = spinful_fermion_basis_1d(L=params.nsites, Nf=(params.nup, params.ndown))


# defines onsite coupling
onsite_interaction = [[1., site, site] for site in range(nsites)]

# these define hopping interaction
hopping = [[1., site, site+1] for site in range(nsites - 1)]
# periodic boundary conditions
if params.bc == "periodic":
    hopping.append([1., nsites - 1, 0])

# onsite hamiltonian just counts the number of spin up and spin down electrons
onsite = hamiltonian([["n|n", onsite_interaction]], [], basis=basis)
# left hopping portion of the hamiltonian allows spin up and spin down electrons to hop left
# (destroying on site i+1 and creating on site i), but it is not hermitian until combined with
# the right hopping portion, so we ignore checks
hop_left = hamiltonian([["+-|", hopping], ["|+-", hopping]], [], basis=basis, check_herm=False, check_symm=False,
                       check_pcon=False)
# right hopping term is just the hermitian conjugate of hop left
hop_right = hop_left.getH()

"""This is the system hamiltonian"""
ham = -params.t0 * (hop_left + hop_right) + params.u * onsite


"""Ground state is the eigenstate of the hamiltonian corresponding to the lowest eigenergy"""
(exact_e,), exact_ground = ham.eigsh(k=1, which='SA')
tf = time()
print("Exact energy of the ground state with U = {}t0 is {}".format(params.u, exact_e))
print("It took", tf - ti, "seconds to calculate")
'''



"""
---------------------------------------
Simulate ground state energy with tenpy
---------------------------------------
"""
model_params = dict(cons_N='N', cons_Sz='Sz', L=params.nsites, t=params.t0,U=params.u)
search = GroundStateSearch(dict(model_class=FermiHubbardModel, model_params=model_params, initial_state_params=dict(method="mps_product_state", product_state=["up", "down"]*(params.nsites//2))))
start = time()
results = search.run()
end = time()
print("Ground state energy: {}".format(results["energy"]))
print("Runtime: {} seconds".format( (end-start) ))