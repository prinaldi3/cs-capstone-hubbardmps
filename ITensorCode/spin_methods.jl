using ITensors
include "evolve.jl"

"""
Get the Hamiltonian as an MPO at time
"""
function get_ham(time, params)
    # get phi at time
    phi = phi_tl(time, params, independent)

    # single particle hamiltonian
    one = OpSum()

    cosphi = cos(phi)
    sinphi = sin(phi)

    for j=1:params.nsites-1
        one += -2*cosphi, "Cdagup", j, "Cup", j+1
        one += -eiphiconj, "Cdagdn", j, "Cdn", j+1
        one += -eiphi, "Cdagup", j+1, "Cup", j
        one += -eiphi, "Cdagdn", j+1, "Cdn", j
    end

    # periodic boundary conditions
    one += -eiphiconj, "Cdagup", params.nsites, "Cup", 1
    one += -eiphiconj, "Cdagdn", params.nsites, "Cdn", 1
    one += -eiphi, "Cdagup", 1, "Cup", params.nsites
    one += -eiphi, "Cdagdn", 1, "Cdn", params.nsites

    # two particle hamiltonian
    two = OpSum()
    for j=1:params.nsites
        two += params.U, "Nup", j, "Ndn", j
    end

    h1 = MPO(one, params.space)
    h2 = MPO(two, params.space)
    H = h1 + h2

    H = splitblocks(linkinds, H)
    return H
end
