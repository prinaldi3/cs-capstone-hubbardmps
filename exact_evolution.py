from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d
import quspin.tools.evolution as evolution
import numpy as np
import exact_evolve as evolve
from tools import Parameters
from multiprocessing import Pool

# system size
N = 10
# energy parameters, in units eV
it = .52
iU = 4 * it

# lattice spacing, in angstroms
ia = 4

# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

# periodic boundary conditions
pbc = False

lat = Parameters(N, iU, it, ia, cycles, iomega0, iF0, pbc)

data = []
for N in [6, 8]:
    for nsteps in [1000, 2000, 3000, 4000, 5000]:
        for uot in [0, .125, .25, .5, 1., 2., 4., 8.]:
            data.append((Parameters(N, uot * it, it, ia, cycles, iomega0, iF0, pbc), nsteps))

def simulate(lat, n_steps):

    """System Evolution Time"""
    start = 0
    stop = 2 * np.pi * cycles / lat.field
    times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

    str = f"U{lat.u}-nsites{lat.nsites}-nsteps{n_steps}"  # for storing expectations
    if lat.pbc:
        str += "-pbc"

    """create basis"""
    basis = spinful_fermion_basis_1d(lat.nsites, Nf=(lat.nsites//2, lat.nsites//2))

    """Create static part of hamiltonian - the interaction b/w electrons"""
    int_list = [[1.0, i, i] for i in range(lat.nsites)]
    static_Hamiltonian_list = [
        ["n|n", int_list]  # onsite interaction
    ]
    # n_j,up n_j,down
    onsite = hamiltonian(static_Hamiltonian_list, [], basis=basis)

    """Create dynamic part of hamiltonian - composed of a left and a right hopping parts"""
    hop = [[1.0, i, i+1] for i in range(lat.nsites-1)]
    if lat.pbc:
        hop.append([1.0, lat.nsites-1, 0])  # periodic BC
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    # c^dag_j,sigma c_j+1,sigma
    hop_left = hamiltonian([["+-|", hop], ["|+-", hop]], [], basis=basis, **no_checks)
    # c^dag_j+1,sigma c_j,sigma
    hop_right = hop_left.getH()

    """Create complete Hamiltonian"""
    H = -lat.t0 * (hop_left + hop_right) + lat.u * onsite

    """get ground state as the eigenstate corresponding to the lowest eigenergy"""
    # print("calculating ground state")
    E, psi_0 = H.eigsh(k=1, which='SA')
    psi_0 = np.squeeze(psi_0)
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    # print("ground state calculated, energy is {:.2f}".format(E[0]))

    # print('evolving system')
    """evolving system, using our own solver for derivatives"""
    psi_t = evolution.evolve(psi_0, 0.0, times, evolve.evolve_psi, f_params=(onsite, hop_left, hop_right, lat))
    psi_t = np.squeeze(psi_t)

    """Calculate Expectation Values"""
    J_expec = evolve.J_expec(psi_t, times, hop_left, hop_right, lat)
    H_expec = evolve.H_expec(psi_t, times, onsite, hop_left, hop_right, lat)

    np.save('./Data/Exact/current-'+str+'.npy', J_expec)
    np.save('./Data/Exact/energy-'+str+'.npy', H_expec)
    np.save('./Data/Exact/times-nsteps{}.npy'.format(n_steps), times)

    print("Done - " + str)

with Pool(80) as pool:
    pool.starmap(simulate, data)
