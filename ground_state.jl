using ITensors

N = 12
# energy parameters, for now these will be prescaled
# t = .52
# U = 1 * t
t = 1
U = 1 * t


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

# two particle hamiltonian
two = OpSum()
for j=1:N
    global two += U, "Nup", j, "Ndn", j
end

h1 = MPO(one, sites)
h2 = MPO(two, sites)

# Prepare initial state MPS
state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_i = productMPS(sites , state)

# Do 10 sweeps of DMRG , gradually
# increasing the maximum MPS
# bond dimension
sweeps = Sweeps(6)
maxdim!(sweeps, 10, 20, 100, 200, 400)
# maxdim!( sweeps ,10 ,20 ,100 ,200 ,400 ,800)
cutoff!(sweeps ,1e-10)
# Run the DMRG algorithm
energy , psi0 = dmrg([h1, h2],psi0_i , sweeps )

@show energy
