using MKL
using ITensors
using LinearAlgebra
include("evolve.jl")

ITensors.Strided.set_num_threads(1)
MKL.BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

# system size
N = 10
# energy parameters, in units eV
it = .52
iU = 0.5 * it

iomega0 = 32.9  # driving (angular) frequency, in THz
factor = 1 / (it * 0.036749323)
omega0 = iomega0 * factor * 0.0001519828442

cycles = 10

U = iU / it

# create the local hilbert space on N sites
sites = siteinds("Electron", N; conserve_qns=true)
H_ground = get_ham(N, sites, 0, U)

# Prepare initial state MPS
istate = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(sites , istate)

# Do 8 sweeps of DMRG , gradually increasing the maximum MPS
# bond dimension, at 12 sites, this gives precision to 7 sig figs
sweeps = Sweeps(8)
maxdim!(sweeps, 10, 20, 100, 200, 400, 600, 800)
cutoff!(sweeps, 1e-10)
# Run the DMRG algorithm
energy, psi0 = @time dmrg(H_ground, psi0_i, sweeps)

@show energy

# imaginary times for evolution
ti = 0im
tf = 2.0im * pi * cycles / omega0
cutoff = 1E-8

# for imaginary evolution, we just want the undriven hamiltonian
ham = get_itensor_ham(N, sites, 0, U)

for nsteps in [1000, 2000, 3000, 4000, 5000], state in ["ground", "neel"]

    io = open("./Data/RK4/im-$state-energy-U$U-nsites$N-nsteps$nsteps.txt", "w")
    close(io)

    energies = zeros(nsteps)

    if state == "ground"
        psi = psi0
    else
        psi = psi0_i
    end

    tau = (tf - ti) / nsteps  # time step
    @show tau
    @show typeof(tau)

    #Time evolution
    @time for step=0:nsteps-1
        curr_time = step * tau
        k1 = -1.0im * tau * apply(ham, psi)
        k2 = -1.0im * tau * apply(ham, psi + 0.5 * k1)
        k3 = -1.0im * tau * apply(ham, psi + 0.5 * k2)
        k4 = -1.0im * tau * apply(ham, psi + k3)
        psi += (1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k4
        psi = get_normalized_MPS!(psi)
        # calculate energy by taking <psi|H|psi>
        local energy = inner(psi, H_ground, psi)
        energies[step + 1] = real(energy)
    end

    io = open("./Data/RK4/im-$state-energy-U$U-nsites$N-nsteps$nsteps.txt", "w")
    for step=1:nsteps
        write(io, "$(energies[step])\n")
    end
    close(io)

end
