using MKL
using ITensors
include("evolve.jl")

ITensors.Strided.set_num_threads(1)
MKL.BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

# system size
N = 10
# energy parameters, in units eV
it = .52

# lattice spacing, in angstroms
ia = 4

# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

# CONVERTING TO ATOMIC UNITS, w/ energy normalized to t
factor = 1 / (it * 0.036749323)
t = 1

omega0 = iomega0 * factor * 0.0001519828442
a = ia * 1.889726125/factor
F0 = iF0 * 1.944689151e-4 * (factor^2)

# create the local hilbert space on N sites
sites = siteinds("Electron", N; conserve_qns=true)

# Prepare initial state MPS
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(sites , state)

for U in [10]

    H_ground = get_ham(N, sites, 0, U)

    # Do 8 sweeps of DMRG , gradually increasing the maximum MPS
    # bond dimension, at 12 sites, this gives precision to 7 sig figs
    sweeps = Sweeps(8)
    maxdim!(sweeps, 10, 20, 100, 200, 400, 400, 600)
    cutoff!(sweeps, 1e-10)
    # Run the DMRG algorithm
    energy, psi0 = @time dmrg(H_ground, psi0_i, sweeps)

    @show energy

    # times for evolution
    nsteps = 1000
    ti = 0
    tf = 2 * pi * cycles / omega0
    tau = (tf - ti) / nsteps  # time step
    cutoff = 1E-8

    psi = psi0

    io = open("./Data/EvolutionTesting/mps-U$U-nsites$N-nsteps$nsteps.txt", "w")
    write(io, "$nsteps\n")
    #Time evolution
    @time for step=0:nsteps-1
        curr_time = step * tau
        phi = phi_tl(curr_time, a, F0, omega0, N)
        psi = apply(get_prop_gates(N, sites, tau, phi, U), psi; cutoff=cutoff)
        # calculate energy by taking <psi|H|psi>
        local current = inner(psi, get_current(N, sites, phi, a), psi)
        write(io, "$current\n")
    end
    close(io)
end


for U in [.5, 1, 2, 5, 10]

    H_ground = get_ham(N, sites, 0, U)

    # Do 8 sweeps of DMRG , gradually increasing the maximum MPS
    # bond dimension, at 12 sites, this gives precision to 7 sig figs
    sweeps = Sweeps(8)
    maxdim!(sweeps, 10, 20, 100, 200, 400, 400, 600)
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

    io = open("./Data/EvolutionTesting/mps-U$U-nsites$N-nsteps$nsteps.txt", "w")
    write(io, "$nsteps\n")
    #Time evolution
    @time for step=0:nsteps-1
        curr_time = step * tau
        phi = phi_tl(curr_time, a, F0, omega0, N)
        psi = apply(get_prop_gates(N, sites, tau, phi, U), psi; cutoff=cutoff)
        # calculate energy by taking <psi|H|psi>
        local current = inner(psi, get_current(N, sites, phi, a), psi)
        write(io, "$current\n")
    end
    close(io)
end
