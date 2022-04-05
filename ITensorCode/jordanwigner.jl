using ITensors

# ITensors.enable_debug_checks()

"""
Mapping is defined by creating two subspaces (Up and Down) and taking the tensor
product (up) x (down)
"""

"""
Defining the local hilbert space for jordan wigner
"""
function ITensors.space(::SiteType"JW"; conserve_qns=false)
    # this will be changed to incorporate quantum numbers
    if conserve_qns
        # return [
        #     QN(("Nf", 2, -1), ("Sz", 2)) => 1
        #     QN(("Nf", 1, -1), ("Sz", 0)) => 1
        #     QN(("Nf", 1, -1), ("Sz", 0)) => 1
        #     QN(("Nf", 0, -1), ("Sz", -2)) => 1
        #     ]
        return 4
    else
        return 4
    end
end

"""
Defining basis states for Jordan Wigner: vacuum, spin down, spin up, full
"""
ITensors.state(::StateName"Vac", ::SiteType"JW") = [0, 0, 0, 1]
ITensors.state(::StateName"Dn", ::SiteType"JW") = [0, 0, 1, 0]
ITensors.state(::StateName"Up", ::SiteType"JW") = [0, 1, 0, 0]
ITensors.state(::StateName"Full", ::SiteType"JW") = [1, 0, 0, 0]

ITensors.op(::OpName"Id", ::SiteType"JW") = [
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1
]

ITensors.op(::OpName"SzUp", ::SiteType"JW") = [
    .5 0 0 0
    0 .5 0 0
    0 0 -.5 0
    0 0 0 -.5
]

ITensors.op(::OpName"NUp", ::SiteType"JW") = [
    1 0 0 0
    0 1 0 0
    0 0 0 0
    0 0 0 0
]


ITensors.op(::OpName"SzDn", ::SiteType"JW") = [
    .5 0 0 0
    0 -.5 0 0
    0 0 .5 0
    0 0 0 -.5
]

ITensors.op(::OpName"SzDn", ::SiteType"JW") = [
    1 0 0 0
    0 0 0 0
    0 0 1 0
    0 0 0 0
]

ITensors.op(::OpName"NUpDn", ::SiteType"JW") = [
    1 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0
]

ITensors.op(::OpName"Nf", ::SiteType"JW") = [
    2 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 0
]

ITensors.op(::OpName"SxUp", ::SiteType"JW") = [
    0 0 .5 0
    0 0 0 .5
    .5 0 0 0
    0 .5 0 0
]

ITensors.op(::OpName"SxDn", ::SiteType"JW") = [
    0 .5 0 0
    .5 0 0 0
    0 0 0 .5
    0 0 .5 0
]

ITensors.op(::OpName"SyUp", ::SiteType"JW") = [
    0 0 -.5im 0
    0 0 0 -.5im
    .5im 0 0 0
    0 .5im 0 0
]

ITensors.op(::OpName"SyDn", ::SiteType"JW") = [
    0 -.5im 0 0
    .5im 0 0 0
    0 0 0 -.5im
    0 0 .5im 0
]

"""
Time independent hamiltonian
"""
function get_ham(sites, nsites, sU)
    H = OpSum()
    for j=1:nsites
        # periodic boundary conditions
        j1 = (j % nsites) + 1
        # one particle term
        H += -2, "SxUp", j, "SxUp", j1
        H += -2, "SyUp", j, "SyUp", j1
        H += -2, "SxDn", j, "SxDn", j1
        H += -2, "SyDn", j, "SyUp", j1
        # two particle term
        H += sU, "NUpDn", j
    end
    return MPO(H, sites)
end

function get_gates(sites, nsites, sU, dt)

    # odd gates (1,2),(3,4),(5,6),...
    ogates = ITensor[]
    # even gates (2,3),(4,5),(6,7),...
    egates = ITensor[]
    for j=1:nsites
        s1 = sites[j]
        #periodic BC
        if j == nsites
            s2 = sites[1]
        else
            s2 = sites[j+1]
        end
        # we have to define the two site operator so contributions
        # of each site will be counted twice
        ul = ur = sU / 2
        # create operator (only scaled parameters are passed in so t is always 1)
        hj = -1 * op("SxUp",s1) * op("SxUp",s2) +
             -1 * op("SxDn",s1) * op("SxDn",s2) +
             -1 * op("SyUp",s1) * op("SyUp",s2) +
             -1 * op("SyDn",s1) * op("SyDn",s2) +
             ul * op("NUpDn", s1) * op("Id",s2) +
             ur * op("Id",s1) * op("NUpDn", s2)
        # odd gate
        if j % 2 == 1
            Gj = exp(-1.0im * dt * hj)
            push!(ogates, Gj)
        # even gate
        else
            Gj = exp(-1.0im * dt / 2 * hj)
            push!(egates, Gj)
        end
    end

    gates = ITensor[]
    append!(gates, egates)
    append!(gates, ogates)
    append!(gates, egates)

    return gates
end

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


thesites = siteinds("JW", N; conserve_qns=true)
ham = get_ham(thesites, N, 0)
psi0_i = productMPS(thesites, [isodd(n) ? "Up" : "Dn" for n=1:N])
sweeps = Sweeps(6)
setmaxdim!(sweeps, 10, 20, 100, 200, 400, 800)
setcutoff!(sweeps, 1e-10)

energy, psi0 = @time dmrg(ham, psi0_i, sweeps)
@show energy

# times for evolution
nsteps = 100000
ti = 0
tf = 2 * pi * cycles / omega0
tau = (tf - ti) / nsteps  # time step
cutoff = 1E-8

gates = get_gates(thesites, N, 0, tau)

psi = psi0
for n=1:nsteps
    global psi = apply(gates, psi;cutoff=cutoff)
    @show inner(psi, ham, psi)
end

# site = siteind("JW")
# o = op("NUpDn", site)
# s = state(site, "Vac")

# o = MPO(o, [s])
# @show (o * state) == state

# function jw_apply(a, b, oa, ob)
#     return oa * a, ob * b
# end
#
# function jw_expectation(a, b, oa, ob)
#     ea = inner(a, oa * a)
#     eb = inner(b, ob * b)
#     return ea * eb
# end
#
# # create the local hilbert spaces
# alphasites = siteinds("S=1/2", N; conserve_qns=true)
# betasites = siteinds("S=1/2", N; conserve_qns=true)
#
# # create different species of spin states with alternating up and down
# alphastate = [isodd(n) ? "Up" : "Dn" for n=1:N]
# betastate = [iseven(n) ? "Up" : "Dn" for n=1:N]
#
# alpha = productMPS(alphasites, alphastate)
# beta = productMPS(betasites, betastate)
#
# oalpha = ITensor[]
# obeta = ITensor[]
#
# for j=1:N
#     sa = alphasites[j]
#     sb = betasites[j]
#     if isodd(j)
#         push!(oalpha, op("Sz", sa))
#     else
#         push!(oalpha, op("Id", sa))
#     end
#     push!(obeta, op("Id", sb))
# end
#
# ea = inner(alpha, apply(oalpha, alpha))
# eb = inner(beta, apply(obeta, beta))
#
# return ea * eb

# @show jw_apply(a, b, opa, opb)
# @show jw_expectation(alpha, beta, a1, b1)
