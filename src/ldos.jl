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
    t0 = (op2 * mps)
    t1 = (H * t0)
    println("n0:", norm(t0))
    println("n1:", norm(t1))
    println("n2:", norm(2 * (H * t1) ))
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
        # println("n",norm(newstate))
        # println("n2", norm(2 * (H * t1) - t0))
        println("sp: ", scalar_product(2 * (H * t1) - t0, newstate) / (norm(2 * (H * t1) - t0) * norm(newstate)))
        #println("dense:", dense(newstate))
        μs[k] = matrix_element(mps, op1, newstate)
        println(μs[k])
        t0 = t1
        t1 = newstate
    end
    return μs
end

function test_ldos()
    N = 20
    D = 10
    Nmax = 20
    mps = randomLCROpenMPS(N, 2, D)
    ham = TensorNetworks.KitaevMPO(N, 1, 1, 0.0, 3)
    states, energy = DMRG(ham, mps)
    println(energy)
    ham2a = -energy * TensorNetworks.DenseIdentityMPO(N, 2) + ham
    ham2b = -energy * TensorNetworks.IdentityMPO(N, 2) + ham
    states2a, energy2a = DMRG(ham2a, mps)
    states2b, energy2b = DMRG(ham2b, mps)
    states3, e = DMRG(-ham2b, mps)
    println(energy2a)
    println(energy2b)
    Emax = -e
    Emin = energy
    #op = MPOsite(sx + sy*im)
    mid = (Emax + Emin) / 2
    w = (Emax - Emin)
    eps = 0.1
    wstar = w #try different choices
    wprime = 1 - eps / 2
    a = wstar / (2 * wprime)
    ham3 = ham / a - (wprime + Emin / a) * TensorNetworks.IdentityMPO(N, 2)
    #ham3d = TensorNetworks.dense(ham3);
    #ham3d2 = ham/a - (wprime + Emin/a)* TensorNetworks.DenseIdentityMPO(N, 2)
    function op(k, a)
        idmpo = MPO([MPOsite(si) for k in 1:N])
        op1 = copy(idmpo)
        op1[k] = MPOsite(sx + a * 1im * sy)
        return op1
    end
    #op_pairs = [(op(k1, 1), op(k1, -1)) for (k1, k2) in Base.product(1:N, 1:N)]
    ops = [(op(k, 1), op(k, -1)) for k in 1:N]
    gammas = jacksonkernel(Nmax - 1)
    # mus = [TensorNetworks.ldos(states, ham3, op1, op2, Nmax=20) for (op1, op2) in ops]
    # functions = [chebyshev(mu, gammas) for mu in mus]

    @time mus1 = TensorNetworks.ldos(states, ham3, ham3, ham3, Nmax=20)
    @time mus2 = TensorNetworks.ldos(states, ham3d, ham3d, ham3d, Nmax=20)
    #coeffs = chebyshev(mus,)
    return mus, gammas, functions, mid, w
end

function jacksonkernel(M)
    [((M - n + 1)cos(π * n / (M + 1)) + sin(π * n / (M + 1))cot(π / (M + 1))) / (M + 1) for n in 0:M]
end

using Polynomials

function chebyshev(mus, gammas)
    coeffs = 2 * mus .* gammas
    coeffs[1] /= 2
    p = ChebyshevT(coeffs)
    return x -> p(x) / (pi * sqrt(1 - x^2))
end