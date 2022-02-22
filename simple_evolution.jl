using MKL
using ITensors

N = 12
# energy parameters, for now these will be prescaled
# t = .52
# U = 1 * t
t = 1
U = 1 * t

ITensors.Strided.set_num_threads(1)
MKL.BLAS.set_num_threads(1)
ITensors.enable_threaded_blocksparse()

# create the local hilbert space on N sites
sites = siteinds("Electron", N; conserve_qns=true)

# H = -t_0 \sum_j,sig \hat{c}^{\dag}_{j,sig} \hat{c}_{j+1,sig} + h.c
#      + U \sum_j \hat{n}_{j, \uparrow} \hat{n}_{j, \downarrow}


# single particle hamiltonian
one = OpSum()
for j=1:N-1
    global one += -t, "Cdagup", j, "Cup", j+1
    global one += -t, "Cdagdn", j, "Cdn", j+1
    global one += -t, "Cdagup", j+1, "Cup", j
    global one += -t, "Cdagdn", j+1, "Cdn", j
end

# periodic boundary conditions
one += -t, "Cdagup", N, "Cup", 1
one += -t, "Cdagdn", N, "Cdn", 1
one += -t, "Cdagup", 1, "Cup", N
one += -t, "Cdagdn", 1, "Cdn", N

# two particle hamiltonian
two = OpSum()
for j=1:N
    global two += U, "Nup", j, "Ndn", j
end

h1 = MPO(one, sites)
h2 = MPO(two, sites)
H = h1 + h2

H = splitblocks(linkinds, H)

# Prepare initial state MPS
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(sites , state)

# Do 8 sweeps of DMRG , gradually increasing the maximum MPS
# bond dimension, at 12 sites, this gives precision to 7 sig figs
sweeps = Sweeps(8)
maxdim!(sweeps, 10, 20, 100, 200, 400, 600)
cutoff!(sweeps, 1e-10)
# Run the DMRG algorithm
energy , psi0 = @time dmrg(H, psi0_i , sweeps )
@show energy

# times for evolution, pretty aribtrary right now
nsteps = 100
ti = 0
tf = .01
tau = (tf - ti) / nsteps
cutoff = 1E-8

# Make gates (1,2),(2,3),(3,4),...
gates = ITensor[]
@time for j=1:N
    s1 = sites[j]
    #periodic BC
    if j == N
        s2 = sites[1]
    else
        s2 = sites[j+1]
    end
    # we have to define the two site operator so contributions
    # of each site will be counted twice
    ul = ur = U / 2
    # create operator
    hj = -t * op("Cdagup",s1) * op("Cup",s2) +
         -t * op("Cdagdn",s1) * op("Cdn",s2) +
         -t * op("Cdagup",s2) * op("Cup",s1) +
         -t * op("Cdagdn",s2) * op("Cdn",s1) +
         ul * op("Nupdn", s1) * op("Id",s2) +
         ur * op("Id",s1) * op("Nupdn", s2)
    Gj = exp(-1.0im * tau/2 * hj)
    push!(gates,Gj)
end

# Include gates in reverse order too
# (N,N-1),(N-1,N-2),...
@time append!(gates,reverse(gates))

#Initialize psi to an MPS
# psi = productMPS(sites, n -> isodd(n) ? "Up" : "Dn")
psi = psi0

#Time evolution
@time for step=1:nsteps
    global psi = apply(gates, psi; cutoff=cutoff)
    # calculate energy by taking <psi|H|psi>
    local energy = inner(psi, H, psi)
    @show energy
end
