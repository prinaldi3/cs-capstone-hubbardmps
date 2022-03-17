using ITensors

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

function RK4()
end
