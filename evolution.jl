using ITensors
using LinearAlgebra

# system size
N = 10
# energy parameters, in units eV
it = .52
iU = 1 * it

# lattice spacing, in angstroms
ia = 4

# pulse parameters
iF0 = 10  # field strength in MV/cm
iomega0 = 32.9  # driving (angular) frequency, in THz
cycles = 10

# CONVERTING TO ATOMIC UNITS, w/ energy normalized to t
factor = 1 / (t * 0.036749323)
t = 1
U = iU / t

omega0 = iomega0 * factor * 0.0001519828442
a = ia * 1.889726125/factor
F0 = iF0 * 1.944689151e-4 * (factor**2)

"""
Get the value of the transform limited pulse phi at time time
\phi(t) = \frac{aF_0}{\omega_0} \sin^2(\frac{\omega_0 t}{2(cycles)}) \sin(\omega_0 t)
params:
lat - scaled lattice constant
strength - scaled field strength
field - scaled driving frequency
cyc - number of cycles
"""
function phi_tl(time, lat, strength, field, cyc)
    return (lat * strength / field) * (sin(field * time / (2*cyc))**2) * sin(field * time)
end

"""
Get the time dependent Hamiltonian at a time corresponding to phi
"""
function get_ham(nsites, space, p, sU)
    # create the local hilbert space on N sites
    sites = siteinds("Electron", nsites; conserve_qns=true)

    # H = -t_0 \sum_j,sig \hat{c}^{\dag}_{j,sig} \hat{c}_{j+1,sig} + h.c
    #      + U \sum_j \hat{n}_{j, \uparrow} \hat{n}_{j, \downarrow}


    # single particle hamiltonian
    one = OpSum()

    eiphi = exp(1.0im * p)
    eiphiconj = conj(eiphi)
    for j=1:nsites-1
        global one += -eiphiconj, "Cdagup", j, "Cup", j+1
        global one += -eiphiconj, "Cdagdn", j, "Cdn", j+1
        global one += -eiphi, "Cdagup", j+1, "Cup", j
        global one += -eiphi, "Cdagdn", j+1, "Cdn", j
    end

    # periodic boundary conditions
    one += -eiphiconj, "Cdagup", nsites, "Cup", 1
    one += -eiphiconj, "Cdagdn", nsites, "Cdn", 1
    one += -eiphi, "Cdagup", 1, "Cup", nsites
    one += -eiphi, "Cdagdn", 1, "Cdn", nsites

    # two particle hamiltonian
    two = OpSum()
    for j=1:nsites
        global two += sU, "Nup", j, "Ndn", j
    end

    h1 = MPO(one, space)
    h2 = MPO(two, space)
    H = h1 + h2

    H = splitblocks(linkinds, H)
    return H
end

"""
Get the time dependent current operator at time corresponding to phi
"""
function get_current(nsites, space, p, sa)
    current = OpSum()

    eiphi = exp(1.0im * p)
    eiphiconj = conj(eiphi)

    # coupling for hopping left and right
    left = -1im * sa * eiphiconj
    right = 1im * sa * eiphi

    for j=1:nsites-1
        current += left, "Cdagup", j, "Cup", j+1
        current += left, "Cdagdn", j, "Cdn", j+1
        current += right, "Cdagup", j+1, "Cup", j
        current += right, "Cdagdn", j+1, "Cdn", j
    end

    # periodic boundary conditions
    current += left, "Cdagup", N, "Cup", 1
    current += left, "Cdagdn", N, "Cdn", 1
    current += right, "Cdagup", 1, "Cup", N
    current += right, "Cdagdn", 1, "Cdn", N

    J = MPO(current, space)

    J = splitblocks(linkinds, J)

    return J
end


"""
Get the gates that represent the propogator at time corresponding to phi
params:
nsites - # of sites in the system
space - local hilbert space with indices to different sites
delta - time step
p - field at time time
sU - scaled U
"""
function get_prop_gates(nsites, space, delta, p, sU)
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j=1:nsites
        s1 = space[j]
        #periodic BC
        if j == N
            s2 = space[1]
        else
            s2 = space[j+1]
        end
        # we have to define the two site operator so contributions
        # of each site will be counted twice
        ul = ur = sU / 2
        # exp(i phi(t)) and exp(-i phi(t))
        eiphi = exp(1.0im * p)
        eiphiconj = conj(eiphi)
        # create operator (only scaled parameters are passed in so t is always 1)
        hj = -eiphiconj * op("Cdagup",s1) * op("Cup",s2) +
             -eiphiconj * op("Cdagdn",s1) * op("Cdn",s2) +
             -eiphi * op("Cdagup",s2) * op("Cup",s1) +
             -eiphi * op("Cdagdn",s2) * op("Cdn",s1) +
             ul * op("Nupdn", s1) * op("Id",s2) +
             ur * op("Id",s1) * op("Nupdn", s2)
        Gj = exp(-1.0im * delta/2 * hj)
        push!(gates,Gj)
    end

    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    append!(gates,reverse(gates))

    return gates
end


ITensors.Strided.set_num_threads(1)
BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

# create the local hilbert space on N sites
sites = siteinds("Electron", N; conserve_qns=true)
H_ground = get_ham(N, sites, 0, U)

# Prepare initial state MPS
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(sites , state)

# Do 6 sweeps of DMRG , gradually
# increasing the maximum MPS
# bond dimension
sweeps = Sweeps(6)
maxdim!(sweeps, 10, 20, 100, 200, 400)
# maxdim!( sweeps ,10 ,20 ,100 ,200 ,400 ,800)
cutoff!(sweeps, 1e-10)
# Run the DMRG algorithm
energy , psi0 = @time dmrg(H_ground, psi0_i , sweeps )

# times for evolution
nsteps = 100
ti = 0
tf = 2 * pi * cycles / omega0
tau = (tf - ti) / nsteps  # time step
cutoff = 1E-8

psi = psi0

#Time evolution
@time for step=0:nsteps
    curr_time = step * tau
    phi = phi_tl(curr_time, a, F0, omega0, N)
    global psi = apply(get_prop_gates(N, sites, tau, phi, U), psi; cutoff=cutoff)
    # calculate energy by taking <psi|H|psi>
    local current = inner(psi, H, psi)
    @show energy
end
