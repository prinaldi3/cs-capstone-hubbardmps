"""
In this file we would like to create the ground state of the half-filled
1D Hubbard model with onsite coupling and compare expectation values
to those generated from the exact state in QuSpin.
"""
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
import numpy as np
from time import time
from tools import Parameters

"""Model parameters"""
nsites = 10
t0 = 0.52
u = 1 * t0
bc = "periodic"

params = Parameters(nsites, u, t0, bc)

basis = spinful_fermion_basis_1d(L=params.nsites, Nf=(params.nup, params.ndown))

