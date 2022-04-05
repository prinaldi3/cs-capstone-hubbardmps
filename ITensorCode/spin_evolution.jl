using ITensors
include("evolve.jl")

# time independent evolution or no
independent = true

# system size
N = 6
# energy parameters, in units eV
it = .52
iU = 0.5 * it

# lattice spacing, in angstroms
ia = 4

# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 6  # cycles of the driving field, CHANGE BACK TO 10

# CONVERTING TO ATOMIC UNITS, w/ energy normalized to t
factor = 1 / (it * 0.036749323)
t = 1
U = iU / it

omega0 = iomega0 * factor * 0.0001519828442
a = ia * 1.889726125/factor
F0 = iF0 * 1.944689151e-4 * (factor^2)

# create the local hilbert space on N sites
sites = siteinds("S=1/2", N; conserve_qns=true)

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
nsteps = 10000
ti = 0
tf = 2 * pi * cycles / omega0
tau = (tf - ti) / nsteps  # time step
cutoff = 1E-8

# for dynamic time stepping
epsilon = 1e-4

psi = psi0

# REMEMBER TO TEST WRITING BEFORE RUNNING TEST SCRIPT

io = open("./Data/DynamicTimeStep/$method-current-U$U-nsites$N.txt", "w")
close(io)
io = open("./Data/DynamicTimeStep/$method-energy-U$U-nsites$N.txt", "w")
close(io)

#Time evolution
times, energies, currents = @time propogation(psi, params, tf, method, tau, epsilon, independent)

io = open("./Data/DynamicTimeStep/$method-current-U$U-nsites$N.txt", "w")
for step=1:len(times)
    write(io, "$(times[step]), $(currents[step])\n")
end
close(io)

io = open("./Data/DynamicTimeStep/$method-current-U$U-nsites$N.txt", "w")
for step=1:len(times)
    write(io, "$(times[step]), $(currents[step])\n")
end
close(io)
