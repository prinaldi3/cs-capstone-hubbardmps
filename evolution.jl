using MKL
using ITensors
using LinearAlgebra
include("evolve.jl")
include("methods.jl")

ITensors.Strided.set_num_threads(1)
MKL.BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

# system size
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

# CONVERTING TO ATOMIC UNITS, w/ energy normalized to t
factor = 1 / (it * 0.036749323)
t = 1
U = iU / it

omega0 = iomega0 * factor * 0.0001519828442
a = ia * 1.889726125/factor
F0 = iF0 * 1.944689151e-4 * (factor^2)

# create the local hilbert space on N sites
sites = siteinds("Electron", N; conserve_qns=true)

# set up parameters struct
params = Parameters(N, sites, U, a, cycles, omega0, F0)

# get ground hamiltonian as an MPO
H_ground = get_ham(0, params)

# Prepare initial state MPS
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(params.space, state)

# Do 8 sweeps of DMRG , gradually increasing the maximum MPS
# bond dimension, at 12 sites, this gives precision to 7 sig figs
sweeps = Sweeps(8)
maxdim!(sweeps, 10, 20, 100, 200, 400, 600, 800)
cutoff!(sweeps, 1e-10)
# Run the DMRG algorithm
energy, psi0 = @time dmrg(H_ground, psi0_i, sweeps)

@show energy

# times for evolution
nsteps = 2000
ti = 0
tf = 2 * pi * cycles / omega0
tau = (tf - ti) / nsteps  # time step
cutoff = 1E-8

psi = psi0

currents = zeros(nsteps)
energies = zeros(nsteps)

#Time evolution
@time for step=0:nsteps-1
    curr_time = step * tau
    phi_t = phi_tl(curr_time, a, F0, omega0, cycles)
    phi_td2 = phi_tl(curr_time + tau / 2, a, F0, omega0, cycles)
    phi_td = phi_tl(curr_time + tau, a, F0, omega0, cycles)
    htd2 = get_itensor_ham(N, sites, phi_td2, U)
    k1 = -1.0im * tau * apply(get_itensor_ham(N, sites, phi_t, U), psi)
    k2 = -1.0im * tau * apply(htd2, psi) + -1.0im * tau / 2 * apply(htd2, k1)
    k3 = -1.0im * tau * apply(htd2, psi) + -1.0im * tau / 2 * apply(htd2, k2)
    k4 = -1.0im * tau * apply(get_itensor_ham(N, sites, phi_td, U), psi + k3)
    global psi += (1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k1
    global psi = (1 / norm(psi)) * psi
    # global psi = apply(get_prop_gates(N, sites, tau, phi, U), psi; cutoff=cutoff, maxdim=800)
    # calculate energy by taking <psi|H|psi>
    local current = inner(psi, get_current(N, sites, phi_td, a), psi)
    currents[step + 1] = real(current)
    local energy = inner(psi, get_ham(N, sites, phi_td, U), psi)
    energies[step + 1] = real(energy)
end

io = open("./Data/RK4Testing/mps-current-U$U-nsites$N-nsteps$nsteps.txt", "w")
for step=1:nsteps
    write(io, "$(currents[step])\n")
end
close(io)

io = open("./Data/RK4Testing/mps-energy-U$U-nsites$N-nsteps$nsteps.txt", "w")
for step=1:nsteps
    write(io, "$(energies[step])\n")
end
close(io)
