#https://journals.aps.org/prb/pdf/10.1103/PhysRevB.83.195115

"""
    ldos()


"""
function ldos(mps, hamiltonian, op1, op2; Nmax=100, prec=1e-8, maxiter=50, shifter=ShiftCenter)
    H = hamiltonian #TODO: implement rescalings
    μs = Vector{eltype(mps[1])}(undef, Nmax)
    #norms = Vector{eltype(mps[1])}(undef, Nmax)
    #println(typeof(dense((op2 * mps)[1])))
    #println(typeof(((op2 * mps)[1]))) #TODO Replace getindex for LazyProduct
    t0 = LCROpenMPS(op2 * mps)
    t1 = LCROpenMPS(H * t0)
    μs[1] = matrix_element(t0, op1, t0)
    μs[2] = matrix_element(t1, op1, t0)
    println(μs[1])
    println(μs[2])
    #t2 = 2*(H*t1) - t0
    #println(2*(H*t1) - t0)
    rec(t1, t0) = norm(2 * (H * t1) - t0) * iterative_compression(2 * (H * t1) - t0, LCROpenMPS(t1), prec, maxiter=maxiter, shifter=shifter)
    for k in 3:Nmax
        #println(rec(t1,t0).scalings)
        #println("2Ht1: ",typeof(2*(H*t1)))
        #println("t0:", typeof(t0))
        #println("diff: ", typeof(2*(H*t1) - t0))
        #println(k)
        #println.(typeof.(t1.states))
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
    D = 20
    mps = randomLCROpenMPS(N, 2, D)
    ham = TensorNetworks.KitaevMPO(N, 1, 1, 0.0, 3)
    states, energy = DMRG(ham, mps)
    println(energy)
    ham2 = -energy * TensorNetworks.DenseIdentityMPO(N, 2) + ham
    states2, energy2 = DMRG(ham2, mps)
    states3, Emax = DMRG(-ham2, mps)
    #op = MPOsite(sx + sy*im)
    ham3 = (ham2 / (-Emax))
    ham3d = dense(ham2 / (-Emax))
    @time mus1 = TensorNetworks.ldos(states, ham3, ham3, ham3)
    @time mus2 = TensorNetworks.ldos(states, ham3d, ham3d, ham3d)
    return mus1, mus2
end

function jacksonkernel(M)
    [((M-n+1)cos(π*n/(M+1)) + sin(π*n/(M+1))cot(π/(M+1)))/(M+1) for n in 0:M]
end

using Polynomials

function chebyshev(mus, gammas)
    coeffs = 2*mus .* gammas
    coeffs[1] /= 2
    p = ChebyshevT(coeffs)
    return x-> p(x)/ (pi*sqrt(1-x^2))
end