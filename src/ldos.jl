#https://journals.aps.org/prb/pdf/10.1103/PhysRevB.83.195115

"""
    ldos()


"""
function ldos(mps, hamiltonian, op1, op2; Nmax = 100, prec = 1e-8, maxiter = 50, shifter = ShiftCenter)
    H = hamiltonian #TODO: implement rescalings
    μs = Vector{eltype(mps[1])}(undef, Nmax)
    #norms = Vector{eltype(mps[1])}(undef, Nmax)
    t0 = op2 * mps
    t1 = H * t0
    μs[1] = matrix_element(t0, op1, t0)
    μs[2] = matrix_element(t1, op1, t0)
    println(μs[1])
    println(μs[2])
    #t2 = 2*(H*t1) - t0
    #println(2*(H*t1) - t0)
    rec(t1, t0) = norm(2 * (H * t1) - t0) * iterative_compression(2 * (H * t1) - t0, LCROpenMPS(t1), prec, maxiter = maxiter, shifter = shifter)
    for k in 3:Nmax
        #println(rec(t1,t0).scalings)
        #println("2Ht1: ",typeof(2*(H*t1)))
        #println("t0:", typeof(t0))
        #println("diff: ", typeof(2*(H*t1) - t0))
        newstate = (rec(t1, t0))
        #println("dense:", dense(newstate))
        μs[k] = matrix_element(mps, op1, newstate)
        println(μs[k])
        t0 = t1
        t1 = newstate
    end
    return μs
end

function test_ldos()
    N = 10
    D = 10
    mps = randomLCROpenMPS(N, 2, D);
    ham = TensorNetworks.KitaevMPO(N, 1, 1, 0.0, 3);
    states, energy = DMRG(ham, mps)
    println(energy)
    ham2 = -energy *TensorNetworks.DenseIdentityMPO(N,2) + ham;
    states2, energy2 = DMRG(ham2, mps)
    #op = MPOsite(sx + sy*im)
    ldos(states, ham2, ham2, ham2)
end