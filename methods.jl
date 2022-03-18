using ITensors

"""
Get the time dependent Hamiltonian as an ITensor at a time corresponding to phi
Parameters:
    time
    params - an instance of the Parameters class
"""
function get_itensor_ham(time, params)

    # get phi at time
    phi = phi_tl(time, params)

    H = ITensor[]

    for j=1:params.nsites
        s1 = parms.space[j]
        #periodic BC
        if j == params.nsites
            s2 = params.space[1]
        else
            s2 = params.space[j+1]
        end
        # we have to define the two site operator so contributions
        # of each site will be counted twice
        ul = ur = params.U / 2
        # exp(i phi(t)) and exp(-i phi(t))
        eiphi = exp(1.0im * phi)
        eiphiconj = conj(eiphi)
        # create operator (only scaled parameters are passed in so t is always 1)
        hj = -eiphiconj * op("Cdagup",s1) * op("Cup",s2) +
             -eiphiconj * op("Cdagdn",s1) * op("Cdn",s2) +
             -eiphi * op("Cdagup",s2) * op("Cup",s1) +
             -eiphi * op("Cdagdn",s2) * op("Cdn",s1) +
             ul * op("Nupdn", s1) * op("Id",s2) +
             ur * op("Id",s1) * op("Nupdn", s2)
        push!(H, hj)
        end
    return H
end

"""
Use second order TEBD to evolve psi(t) to psi (t + dt)
"""
function TEBD(psi, dt, time, params)

    phi = phi_tl(time, params)

    # odd gates (1,2),(3,4),(5,6),...
    ogates = ITensor[]
    # even gates (2,3),(4,5),(6,7),...
    egates = ITensor[]
    for j=1:params.nsites
        s1 = params.space[j]
        #periodic BC
        if j == params.nsites
            s2 = params.space[1]
        else
            s2 = params.space[j+1]
        end
        # we have to define the two site operator so contributions
        # of each site will be counted twice
        ul = ur = params.U / 2
        # exp(i phi(t)) and exp(-i phi(t))
        eiphi = exp(1.0im * phi)
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

    return apply(gates, psi)
end

function RK4(psi, dt, time, params)
    k1 = -1.0im * dt * apply(get_itensor_ham(time, params), psi)
    k2 = -1.0im * dt * apply(get_itensor_ham(time + dt / 2, params), psi + 0.5 * k1)
    k3 = -1.0im * dt * apply(get_itensor_ham(time + dt / 2, params), psi + 0.5 * k2)
    k4 = -1.0im * dt * apply(get_itensor_ham(time + dt, params), psi + k3)
    next = psi + (1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k4
    return get_normalized_MPS!(next)
end

function RK2(psi, dt, time, params)
    k1 = -1.0im * dt * apply(get_itensor_ham(time, params), psi)
    k2 = -1.0im * dt * apply(get_itensor_ham(time + dt, params), psi + k1)
    next = psi + 0.5 * (k1 + k2)
    return get_normalized_MPS!(next)
end
