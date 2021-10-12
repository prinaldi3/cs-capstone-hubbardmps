"""
In this file we would like to create the ground state of the half-filled
1D Hubbard model with onsite coupling and compare expectation values
to those generated from the exact state in QuSpin.
"""

nsites = 10
t0 = 0.52
u = 1 * t0
bc = "periodic"
