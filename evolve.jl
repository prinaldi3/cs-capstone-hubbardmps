using ITensors

"""
Struct containing scaled parameters
Paramters:
    nsites - number of sites in the system
    space - local hilbert space
    U - value of U normalized to t_0
    a - scaled lattice constant
    cycles - number of cycles of the driving field
    field - scaled angular frequency of the field
    strength - scaled field strength
"""
struct Parameters
    nsites
    space
    U
    a
    cycles
    field
    strength
end

function get_normalized_MPS!(M::MPS; (lognorm!)=[])
  c = ortho_lims(M)
  lognorm_M = lognorm(M)
  push!(lognorm!, lognorm_M)
  z = exp(lognorm_M / length(c))
  # XXX: this is not modifying `M` in-place.
  # M[c] ./= z
  for n in c
    M[n] ./= z
  end
  return M
end


"""
Get the value of the transform limited pulse phi at time time
\phi(t) = \frac{aF_0}{\omega_0} \sin^2(\frac{\omega_0 t}{2(cycles)}) \sin(\omega_0 t)
Parameters:
    time - current time
    params - an instance of Parameters struct
    independent - boolean that is True if we are evolving with a time independent
                  Hamiltonian, and False otherwise
"""
function phi_tl(time, params, independent)
    if independent
        return 0
    else
        return (params.a * params.strength / params.field) \
        * (sin(params.field * time / (2*params.cycles))^2) * sin(params.field * time)
    end
end

"""
Function defines the difference b/w two mps
1 is orthogonal
0 is parallel
"""
function difference(a::MPS, b::MPS)
    return 1 - inner(a, b)
end

"""
Propogates ground from 0 to tf using method to propogate by an adaptive timestep
Parameters:
    ground - ground state (MPS)
    tf - final time
    method - the function used to propogate a wavefunction over single dt
    dti - initial guess for timestep
    epsilon - total change allowed in a single time step
    independent - boolean that is True if we are evolving with a time independent
                  Hamiltonian, and False otherwise
"""
function propogation(ground::MPS, params, tf, method, dti, epsilon, independent)
    # error measures (en is for current step, en1 is for previous step)
    en = en1 = 0.0
    time = 0.0
    # copy states
    psi = deepcopy(ground)
    # set dt and previous dt
    dt = pdt = dti

    # initialize vectors for saving times and expectation values
    times = [0]
    energies = [inner(ground, get_ham(0, params), ground)]
    currents = [inner(ground, get_current(0, params), ground)]

    # run for the entire time interval
    while time < tf

        # get the MPS at time + dt
        next_psi = method(psi, dt, time, params, independent)

        # calculate difference between current and next psi
        en = difference(psi, next_psi)

        # run propogation while the difference is greater than acceptable error
        while en > epsilon
            # adjust time step
            dt *= epsilon / en
            # get the next MPS and calculate difference
            next_psi = method(psi, dt, time, params, independent)
            en = difference(psi, next_psi)
        end

        # incremement time and add it to the array
        time += dt
        times = vcat(times, [time])

        # accept wavefunction
        psi = deepcopy(next_psi)

        # calculate expectations
        energies = vcat(energies, [inner(psi, get_ham(time, params, independent))])
        currents = vcat(currents, [inner(psi, get_current(time, params, independent))])

        en1 = (en1 > 0) ? en1 : en

        # adjust for next time step
        # https://www.sciencedirect.com/science/article/pii/S0377042705001123
        ndt = dt * ((epsilon**2 * dt)/ (en * en1 * pdt))^(1/12)

        # update values for next iteration e_{n-1} -> e_n, dt_{n-1} = dt_n,
        # dt_n -> dt_{n+1}
        en1 = en
        pdt = dt
        dt = ndt
    end
    return times, energies, currents
end

"""
Get the time dependent Hamiltonian as an MPO at a time corresponding to phi
Parameters:
    time
    params - an instance of the Parameters class
"""
function get_ham(time, params, independent)
    # H = -t_0 \sum_j,sig \hat{c}^{\dag}_{j,sig} \hat{c}_{j+1,sig} + h.c
    #      + U \sum_j \hat{n}_{j, \uparrow} \hat{n}_{j, \downarrow}

    # get phi at time
    phi = phi_tl(time, params, independent)

    # single particle hamiltonian
    one = OpSum()

    eiphi = exp(1.0im * phi)
    eiphiconj = conj(eiphi)
    for j=1:params.nsites-1
        one += -eiphiconj, "Cdagup", j, "Cup", j+1
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

"""
Get the time dependent current operator as an MPO at time corresponding to phi
Parameters:
    time
    params - an instance of the Parameters class
"""
function get_current(time, params, independent)

    # get phi at time
    phi = phi_tl(time, params, independent)

    current = OpSum()

    eiphi = exp(1.0im * phi)
    eiphiconj = conj(eiphi)

    # coupling for hopping left and right
    left = -1im * sa * eiphiconj
    right = 1im * sa * eiphi

    for j=1:params.nsites-1
        current += left, "Cdagup", j, "Cup", j+1
        current += left, "Cdagdn", j, "Cdn", j+1
        current += right, "Cdagup", j+1, "Cup", j
        current += right, "Cdagdn", j+1, "Cdn", j
    end

    # periodic boundary conditions
    current += left, "Cdagup", params.nsites, "Cup", 1
    current += left, "Cdagdn", params.nsites, "Cdn", 1
    current += right, "Cdagup", 1, "Cup", params.nsites
    current += right, "Cdagdn", 1, "Cdn", params.nsites

    J = MPO(current, params.space)

    J = splitblocks(linkinds, J)

    return J
end
