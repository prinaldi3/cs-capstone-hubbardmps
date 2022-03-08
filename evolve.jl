using ITensors

# """
# Get the value of the transform limited pulse phi at time time
# \phi(t) = \frac{aF_0}{\omega_0} \sin^2(\frac{\omega_0 t}{2(cycles)}) \sin(\omega_0 t)
# params:
# lat - scaled lattice constant
# strength - scaled field strength
# field - scaled driving frequency
# cyc - number of cycles
# """
function phi_tl(time, lat, strength, field, cyc)
    return (lat * strength / field) * (sin(field * time / (2*cyc))^2) * sin(field * time)
end

# """
# Get the time dependent Hamiltonian at a time corresponding to phi
# """
function get_ham(nsites, space, p, sU)
    # H = -t_0 \sum_j,sig \hat{c}^{\dag}_{j,sig} \hat{c}_{j+1,sig} + h.c
    #      + U \sum_j \hat{n}_{j, \uparrow} \hat{n}_{j, \downarrow}


    # single particle hamiltonian
    one = OpSum()

    eiphi = exp(1.0im * p)
    eiphiconj = conj(eiphi)
    for j=1:nsites-1
        one += -eiphiconj, "Cdagup", j, "Cup", j+1
        one += -eiphiconj, "Cdagdn", j, "Cdn", j+1
        one += -eiphi, "Cdagup", j+1, "Cup", j
        one += -eiphi, "Cdagdn", j+1, "Cdn", j
    end

    # periodic boundary conditions
    one += -eiphiconj, "Cdagup", nsites, "Cup", 1
    one += -eiphiconj, "Cdagdn", nsites, "Cdn", 1
    one += -eiphi, "Cdagup", 1, "Cup", nsites
    one += -eiphi, "Cdagdn", 1, "Cdn", nsites

    # two particle hamiltonian
    two = OpSum()
    for j=1:nsites
        two += sU, "Nup", j, "Ndn", j
    end

    h1 = MPO(one, space)
    h2 = MPO(two, space)
    H = h1 + h2

    H = splitblocks(linkinds, H)
    return H
end

# """
# Get the time dependent current operator at time corresponding to phi
# """
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


# """
# Get the gates that represent the propogator at time corresponding to phi
# Second order TEBD
# params:
# nsites - # of sites in the system
# space - local hilbert space with indices to different sites
# delta - time step
# p - field at time time
# sU - scaled U
# """
function get_prop_gates(nsites, space, delta, p, sU)
    # odd gates (1,2),(3,4),(5,6),...
    ogates = ITensor[]
    # even gates (2,3),(4,5),(6,7),...
    egates = ITensor[]
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
        # odd gate
        if j % 2 == 1
            Gj = exp(-1.0im * delta * hj)
            push!(ogates, Gj)
        # even gate
        else
            Gj = exp(-1.0im * delta/2 * hj)
            push!(egates, Gj)
        end
    end

    gates = ITensor[]
    append!(gates, egates)
    append!(gates, ogates)
    append!(gates, egates)

    return gates
end
