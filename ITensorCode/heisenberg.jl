using ITensors

N = 100
cutoff = 1E-8
# tau = 0.1
ttotal = 241

# Compute the number of steps to do
# Nsteps = Int(ttotal/tau)

# Make an array of 'site' indices
s = siteinds("S=1/2",N;conserve_qns=true)

H = OpSum()
for j=1:N-1
    global H += 1, "Sz", j, "Sz", j+1
    global H += 0.5, "S+", j, "S-", j+1
    global H += 0.5, "S-", j, "S+", j+1
end
H = MPO(H, s)

# Initialize psi to be a product state (alternating up and down)
psi0_i = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

sweeps = Sweeps(6)
maxdim!(sweeps, 10, 20, 100, 200, 400)
cutoff!(sweeps, 1e-10)
# Run the DMRG algorithm
e, psi0 = @time dmrg(H, psi0_i, sweeps)
@show e

for tau in [.1]
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j=1:N-1
        s1 = s[j]
        s2 = s[j+1]
        hj =       op("Sz",s1) * op("Sz",s2) +
             1/2 * op("S+",s1) * op("S-",s2) +
             1/2 * op("S-",s1) * op("S+",s2)
        Gj = exp(-1.0im * tau/2 * hj)
        push!(gates,Gj)
    end
    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    append!(gates,reverse(gates))

    # Compute the number of steps to do
    Nsteps = Int(ttotal/tau)

    psi = psi0

    t = 0.0

    io = open("./Data/HeisenbergTest/nsteps$Nsteps.txt", "w")

    # Do the time evolution by applying the gates
    # for Nsteps steps and printing <Sz> on site c
    for step=1:Nsteps
        psi = apply(gates, psi; cutoff=cutoff)
        t += tau
        energy = real(inner(psi, H, psi))
        write(io, "$t, $energy\n")
    end
    close(io)
end
