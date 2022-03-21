using Test, TensorNetworks, TensorOperations, LinearAlgebra, KrylovKit

@testset "Pauli" begin
    s = 1 / 2
    @test norm(Sx(s) * Sy(s) - Sy(s) * Sx(s) - 1im * Sz(s)) < 1e-14
    @test norm(Sy(s) * Sz(s) - Sz(s) * Sy(s) - 1im * Sx(s)) < 1e-14
    s = 1
    @test norm(Sx(s) * Sy(s) - Sy(s) * Sx(s) - 1im * Sz(s)) < 1e-14
    @test norm(Sy(s) * Sz(s) - Sz(s) * Sy(s) - 1im * Sx(s)) < 1e-14
    s = 3 / 2
    @test norm(Sx(s) * Sy(s) - Sy(s) * Sx(s) - 1im * Sz(s)) < 1e-14
    @test norm(Sy(s) * Sz(s) - Sz(s) * Sy(s) - 1im * Sx(s)) < 1e-14
end

@testset "Types" begin
    mps = randomUMPS(ComplexF64, 2, 2, 1)
    @test mps isa UMPS
    @test mps isa UMPS{ComplexF64}

    @test eltype(mps) == OrthogonalLinkSite{ComplexF64}
    @test eltype(mps[1]) == ComplexF64
    @test TensorNetworks.numtype(mps) == ComplexF64
end

@testset "Gate" begin
    op = rand(Float64, 2, 2)
    g = Gate(op)
    for k1 in 1:2
        for k2 in 1:2
            @test g[k1, k2] == op[k1, k2]
        end
    end

    @test typeof(g) == GenericSquareGate{Float64,2}
    @test complex(typeof(g)) == GenericSquareGate{ComplexF64,2}

    H = (op + op') / 2
    U = exp(H * 1.0im)
    @test ishermitian(op) == ishermitian(g)
    @test ishermitian(Gate(Matrix(H)))
    @test isunitary(op) == isunitary(g)
    @test isunitary(Gate(U))

    id = IdentityGate(1)
    z = rand(ComplexF64)
    @test data(z * id) == z
    @test data(id * z) == z
    @test data(z * g) ≈ z * data(g)
    @test data(g * z) ≈ z * data(g)

    @test g + g ≈ 2 * g ≈ data(g) + data(g)
    @test id + id ≈ 2 * id
    @test g + z * id ≈ data(g) + z * one(data(g))
    @test z * id + g ≈ data(g) + z * one(data(g))

    @test data(exp(z * id)) ≈ exp(z)
    @test data(exp(g)) ≈ exp(data(g))

    @test id' == id
    @test data((z * id)') ≈ z'

    @test Hermitian(z * g) ≈ Gate((z * op)' + (z * op)) / 2

    site = qubit(rand(), rand())
    expval = vec(data(site))' * op * vec(data(site))
    @test expectation_value([site], g) ≈ expval
    g2 = Gate(TensorNetworks.gate(kron(op, op), 2))
    @test expectation_value([site, site], g2) ≈ expval^2

    @test expectation_value([site], z * IdentityGate(1)) ≈ z
    @test expectation_value([site, site], z * IdentityGate(2)) ≈ z
    @test expectation_value([site, site, site], z * IdentityGate(3)) ≈ z

    # D = 10;
    # d = 2;
    # site = randomGenericSite(D,d,D);

    d = 4
    mat = rand(ComplexF64, d, d)
    gate = Gate(mat)
    @test data(gate) ≈ mat
    @test data(gate') ≈ mat'
    @test data(gate * gate) ≈ mat * mat


    tens = rand(ComplexF64, 2, 2, 2, 2)
    gate = Gate(tens)
    @test data(gate) ≈ tens
    @tensor tensc[:] := conj(tens[-3, -4, -1, -2])
    @test data(gate') ≈ tensc
    @tensor t2[:] := tens[-1, -2, 1, 2] * tens[1, 2, -3, -4]
    @test data(gate * gate) ≈ t2

    mat = rand(ComplexF64, d, d)
    mataux = kron(Matrix{ComplexF64}(I, d, d), mat)
    gate = Gate(mat)
    gateaux = TensorNetworks.auxillerate(gate)
    @test data(gateaux) ≈ mataux

    @tensor tens[:] := mat[-1, -3] * mat[-2, -4]
    @tensor tensaux[:] := mataux[-1, -3] * mataux[-2, -4]
    gateaux = TensorNetworks.auxillerate(Gate(tens))
    @test data(gateaux) ≈ tensaux

end

@testset "MPOsite" begin
    id = IdentityMPOsite(2)
    @test data(id)
    z = rand(ComplexF64)
    zid = z * id
    @test data(zid) == z
    @test zid[1, 1, 1, 1] == z
    @test zid[1, 1, 2, 1] == 0
    @test zid[1, 2, 2, 1] == z
    @test id * z == zid
    @test transpose(zid) == zid
    @test conj(zid) == conj(z) * id
    @test id' == id
    @test zid' == conj(z) * id

    DL1, DL2, DR1, DR2, d = rand(2:10, 5)
    site1 = randomGenericSite(DL1, d, DR1)
    site2 = randomGenericSite(DL2, d, DR2)
    T0 = Matrix(transfer_matrix(site1, site2), DL1 * DL2, (DR1, DR2))
    Tid = Matrix(transfer_matrix(site1, id, site2), DL1 * DL2, (DR1, DR2))
    @test T0 == Tid
    @test Matrix(transfer_matrix(site1), DL1^2, (DR1, DR1)) == Matrix(transfer_matrix(site1, id), DL1^2, (DR1, DR1))
    @test z * T0 == Matrix(transfer_matrix(site1, zid, site2), DL1 * DL2, (DR1, DR2))


    DL12, DL22, DR12, DR22 = rand(2:10, 4)
    site12 = site1 + randomGenericSite(DL12, d, DR12)
    site22 = site2 + randomGenericSite(DL22, d, DR22)

    T0 = Matrix(transfer_matrix(site1, site22), (DL1) * (DL2 + DL22), blocksizes([DR1], [DR2, DR22]))
    Tid = Matrix(transfer_matrix(site1, id, site22), (DL1) * (DL2 + DL22), blocksizes([DR1], [DR2, DR22]))
    @test T0 == Tid
    @test z * T0 == Matrix(transfer_matrix(site1, zid, site22), (DL1) * (DL2 + DL22), blocksizes([DR1], [DR2, DR22]))

    T0 = Matrix(transfer_matrix(site12, site22), (DL1 + DL12) * (DL2 + DL22), blocksizes([DR1, DR12], [DR2, DR22]))
    Tid = Matrix(transfer_matrix(site12, id, site22), (DL1 + DL12) * (DL2 + DL22), blocksizes([DR1, DR12], [DR2, DR22]))
    @test T0 == Tid
    @test Matrix(transfer_matrix(site12), (DL1 + DL12)^2, blocksizes([DR1, DR12], [DR1, DR12])) == Matrix(transfer_matrix(site12, id), (DL1 + DL12)^2, blocksizes([DR1, DR12], [DR1, DR12]))
    @test z * T0 == Matrix(transfer_matrix(site12, zid, site22), (DL1 + DL12) * (DL2 + DL22), blocksizes([DR1, DR12], [DR2, DR22]))
end

@testset "MPO" begin
    N = 10
    function test(d)
        id = IdentityMPO(N, d)
        @test length(id) == N
        z = rand(ComplexF64)
        zid = z * id
        @test zid.data == z
        @test zid == id * z

        idsite = IdentityMPOsite(d)
        @test idsite == id[floor(Int, N / 2)]
        @test z^(1 / N) * idsite == zid[floor(Int, N / 2)]
        D = 5
        mps = randomOpenMPS(N, d, D)
        T0 = Matrix(transfer_matrix(mps))
        @test T0 == Matrix(transfer_matrix(mps, id))
        @test z * T0 ≈ Matrix(transfer_matrix(mps, zid))
    end
    test(2)
    test(3)
    @test TensorNetworks.DenseIdentityMPO(10, 2) == TensorNetworks.dense(IdentityMPO(10, 2))


    #LCROpenMPS()
end


@testset "Environment" begin
    N = 10
    d = 2
    mps = randomOpenMPS(N, d, 1)
    env = environment(mps)
    @test length(env.L) == N == length(env.R)
    @test env.L ≈ fill([1.0], N) ≈ env.R

    D = 5
    mps = randomOpenMPS(N, d, D)
    env = environment(mps)
    site = randomGenericSite(D, d, D)
    mid = floor(Int, N / 2)
    update_environment!(env, site, mid)
    TL = transfer_matrix(site, :left)
    TR = transfer_matrix(site, :right)
    @test env.R[mid-1] ≈ TL * env.R[mid]
    @test env.L[mid+1] ≈ TR * env.L[mid]

    D2 = 10
    mps2 = randomOpenMPS(N, d, D2)
    env = environment(mps2, mps)
    @test length(env.R[mid]) == D2 * D == length(env.L[mid])
    site2 = randomGenericSite(D2, d, D2)
    update_environment!(env, site2, site, mid)
    TL = transfer_matrix(site2, site, :left)
    TR = transfer_matrix(site2, site, :right)
    @test env.R[mid-1] ≈ TL * env.R[mid]
    @test env.L[mid+1] ≈ TR * env.L[mid]

    mpo = IsingMPO(N, 1, 1, 1)
    Dmpo = size(mpo[mid], 4)
    env = environment(mps2, mpo, mps)
    @test length(env.R[mid]) == D2 * D * Dmpo == length(env.L[mid])
    update_environment!(env, site2, mpo[mid], site, mid)
    TL = transfer_matrix(site2, mpo[mid], site, :left)
    TR = transfer_matrix(site2, mpo[mid], site, :right)
    @test env.R[mid-1] ≈ TL * env.R[mid]
    @test env.L[mid+1] ≈ TR * env.L[mid]
end

@testset "Canonicalize" begin
    N = 10
    c = 5
    Γ = randomOpenMPS(N, 2, 5).Γ

    L, r = TensorNetworks.to_left_orthogonal(Γ[1], method = :qr)
    @test isleftcanonical(L)
    @test TensorNetworks.data(Γ[1]) ≈ TensorNetworks.data(L * r)
    L, r = TensorNetworks.to_left_orthogonal(Γ[1], method = :svd)
    @test isleftcanonical(L)

    R, l = TensorNetworks.to_right_orthogonal(Γ[1], method = :qr)
    @test isrightcanonical(R)
    @test TensorNetworks.data(Γ[1]) ≈ TensorNetworks.data(l * R)
    R, l = TensorNetworks.to_right_orthogonal(Γ[1], method = :svd)
    @test isrightcanonical(R)

    Γc = TensorNetworks.to_left_right_orthogonal(Γ, center = 5)
    for k in 1:c-1
        @test isleftcanonical(Γc[k])
    end
    @test norm(Γc[c]) ≈ 1
    for k in c+1:N
        @test isrightcanonical(Γc[k])
    end
end

@testset "LCROpenMPS" begin
    N = 5
    mps = randomLCROpenMPS(N, 2, 5)
    for n in 0:N+1
        mps = canonicalize(mps, center = n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0 < n < N + 1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end
    for n in 0:N+1
        set_center!(mps, n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0 < n < N + 1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end

    mps = randomLCROpenMPS(N, 2, 5, purification = true)
    for n in 0:N+1
        mps = canonicalize(mps, center = n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0 < n < N + 1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end
end

@testset "Conversion" begin
    # N = 10
    # mps = canonicalize(randomLCROpenMPS(N,2,5));
    # @test scalar_product(mps',mps) ≈ 1
    # mps2 = OpenMPS(mps);
    # @test scalar_product(mps',mps2) ≈ 1
    # mps3 = LCROpenMPS(mps2);
    # @test scalar_product(mps2',mps3) ≈ 1
    # @test scalar_product(mps',mps3) ≈ 1
end

@testset "Transfer" begin
    D = 10
    d = 2
    R = randomRightOrthogonalSite(D, d, D)
    L = randomLeftOrthogonalSite(D, d, D)
    LR = randomOrthogonalLinkSite(D, d, D)
    id = Matrix{ComplexF64}(I, D, D)
    #idvec = vec(id)
    @test id ≈ transfer_matrix(R) * id
    @test id ≈ transfer_matrix(L, :right) * id
    @test id ≈ transfer_matrix(LR, :right) * id
    @test id ≈ transfer_matrix(LR, :left) * id
    function testsite(site)
        D = size(site, 1)
        d = size(site, 2)
        T = transfer_matrix(site, :left)

        #@test size(T) == (D^2,D^2)
        @test Matrix(T') ≈ Matrix(T)'
        @test transpose(Matrix(T)) ≈ Matrix(transfer_matrix(site, :right))

        z = rand(ComplexF64)
        @test Matrix(T) ≈ Matrix(transfer_matrix(site, IdentityGate(1)))
        @test z * Matrix(T) ≈ Matrix(transfer_matrix(site, z * IdentityGate(1)))

        T1 = transfer_matrix(site, MPOsite(sz))
        #@test size(T1) == (D^2,D^2)
        @test Matrix(T1') ≈ Matrix(T1)'

        g2 = Gate(TensorNetworks.gate(kron(sz, sz), 2))
        T2 = prod(transfer_matrices([site, site], g2))
        @test Matrix(T2) ≈ Matrix(T1 * T1)
        @test transpose(Matrix(T2)) ≈ Matrix(prod(transfer_matrices([site, site], g2, :right)))

        g3 = Gate(TensorNetworks.gate(kron(sz, sz, sz), 3))
        T3 = prod(transfer_matrices([site, site, site], g3))
        @test Matrix(T3) ≈ Matrix(T1 * T1 * T1)
        @test Matrix(T3) ≈ Matrix(prod(transfer_matrices([site, site, site], Gate.([sz, sz, sz]))))
        @test Matrix(T3) ≈ Matrix(prod(transfer_matrices([site, site, site], Gate.([sz, sz, sz]), [site, site, site])))
        @test transpose(Matrix(T3)) ≈ Matrix(prod(transfer_matrices([site, site, site], g3, :right)))

        g4 = Gate(TensorNetworks.gate(kron(sz, sz, sz, sz), 4))
        T4 = prod(transfer_matrices([site, site, site, site], g4))
        @test Matrix(T4) ≈ Matrix(T1 * T1 * T1 * T1)
        @test transpose(Matrix(T4)) ≈ Matrix(prod(transfer_matrices([site, site, site, site], g4, :right)))

        T1 = transfer_matrix(site, MPOsite(sz))
        opsum = MPOsite(sz) + MPOsite(sz)
        MT1 = Matrix(T1)
        @test [MT1 zeros(size(MT1)); zeros(size(MT1)) MT1] ≈ Matrix(transfer_matrix(site, opsum, site))
    end
    site = randomGenericSite(D, d, D)
    testsite(site)
    T = transfer_matrix(site, :left)
    @test Matrix(T) ≈ Matrix(T, D^2, (D, D))

    testsite(site + site)
    T = transfer_matrix(site + site, :left)
    #@test Matrix(T) ≈ Matrix(T, 4 * D^2, blocksizes([D, D], [D, D])) #This test fails when using test, but not in the REPL

    site = randomGenericSite(50, d, 50)
    testsite(site)
    T = transfer_matrix(site, :left)
    @test Matrix(T) ≈ Matrix(T, 50^2, (50, 50))
end

@testset "Compression" begin
    mps = canonicalize(randomOpenMPS(7, 2, 5))
    ΓL = mps[3]
    ΓR = mps[4]
    ΓL2, ΓR2, err = compress(ΓL, ΓR, mps.truncation)
    T = transfer_matrix(ΓL, :left) * transfer_matrix(ΓR, :left)
    T2 = transfer_matrix(ΓL2, :left) * transfer_matrix(ΓR2, :left)
    @test ΓL.Λ1 ≈ ΓL2.Λ1 && ΓL.Λ2 ≈ ΓL2.Λ2 && ΓR.Λ2 ≈ ΓR2.Λ2 && Matrix(T2) ≈ Matrix(T)

    thetaL, thetaR, phiL, phiR = 2 * pi * rand(4)
    ΓL = qubit(thetaL, phiL)
    ΓR = qubit(thetaR, phiR)
    @test norm(ΓL) ≈ norm(ΓR) ≈ 1

    gL = exp(1im * rand(4)' * [si, sx, sy, sz])
    gR = exp(1im * rand(4)' * [si, sx, sy, sz])
    L = gL * vec(data(ΓL))
    R = gR * vec(data(ΓR))

    g = Gate(TensorNetworks.gate(kron(gR, gL), 2))
    ΓL, S, ΓR, err = apply_two_site_gate(ΓL, ΓR, g, mps.truncation)
    @test err < 1e-16
    @test data(S) ≈ [1]
    #@test vec(data(ΓL)) ≈ L && vec(data(ΓR)) ≈ R
    @tensor ΓLR[:] := data(ΓL)[-1, -2, 1] * data(ΓR)[1, -3, -4]
    @tensor LR[:] := reshape(L, 1, 2, 1)[-1, -2, 1] * reshape(R, 1, 2, 1)[1, -3, -4]
    @test ΓLR ≈ LR

end


@testset "TEBD" begin
    d = 4
    mat = rand(ComplexF64, d, d)
    g = Gate(mat)
    expmat = exp(1im * mat)
    id = one(mat)

    layers = TensorNetworks.st1gates(0, [g])
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.st1gates(1, [g])
    @test data(layers[1][1]) ≈ expmat

    layers = TensorNetworks.st2gates(0, [g])
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.st2gates(1, [g])
    @test data(*([l[1] for l in layers[1:2:end]]...)) ≈ expmat

    layers = TensorNetworks.frgates(0, [g])
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.frgates(1, [g])
    @test data(*([l[1] for l in layers[1:2:end]]...)) ≈ expmat
end

@testset "Imaginary TEBD" begin
    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 10
    ham = isingHamGates(Nchain, 1, 1, 0)
    hamMPO = IsingMPO(Nchain, 1, 1, 0)
    mps = canonicalize(identityOpenMPS(Nchain, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states, betas = get_thermal_states(mps, ham, 30, 0.1, order = 2)
    energy = expectation_value(states[1], hamMPO)
    @test abs(energy / (Nchain - 1) + 4 / π) < 1 / Nchain

    ham = isingHamGates(Nchain, 1, 1, 0)[2:3]
    mps = canonicalize(identityUMPS(2, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states, betas = get_thermal_states(mps, ham, 30, 0.1, order = 2)
    energy = (expectation_value(states[1], ham[1], 1) + expectation_value(states[1], ham[2], 2)) / 2
    @test abs(energy + 4 / π) < 1 / Nchain
end

@testset "DMRG" begin
    #Test a few low lying eigenstates of a simple Ising
    Nchain = 5
    Dmax = 10
    ham = IsingMPO(Nchain, 1, 0, 0)
    mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax))
    states, energies = eigenstates(ham, mps, 5; precision = 1e-8)
    @test sort(energies) ≈ -[Nchain - 1, Nchain - 1, Nchain - 3, Nchain - 3, Nchain - 3]

    Nchain = 10
    Dmax = 20
    h, g = (0.226579, 0.988821)
    ham = IsingMPO(Nchain, 1, h, g)
    hammat = Matrix(ham)
    mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax))
    energiesED, _ = eigsolve(hammat, 4, :SR)
    states, energies = eigenstates(ham, mps, 4; precision = 1e-8, shifter = SubspaceExpand(1.0), maxsweeps = 20)
    @test sort(energies) ≈ energiesED[1:4]

    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 30
    ham = IsingMPO(Nchain, 1, 1, 0)
    mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax))
    states, energies = eigenstates(ham, mps, 5; precision = 1e-8, shifter = SubspaceExpand(1.0), maxsweeps = 20)
    @test abs(energies[1] / (Nchain) + 4 / π) < 1 / Nchain

    states, energies = TensorNetworks.eigenstates2(ham, mps, 5; precision = 1e-8, maxsweeps = 20)
    @test abs(energies[1] / (Nchain) + 4 / π) < 1 / Nchain

end

@testset "UMPS expectation values" begin
    theta = 2 * pi * rand()
    phi = 2 * pi * rand()
    h = rand()
    g = rand()
    mps = TensorNetworks.productUMPS(theta, phi)
    hammpo = MPO(IsingMPO(5, 1, h, g)[3])
    hamgates = isingHamGates(5, 1, h, g)[2:3]
    E = expectation_value(mps, hamgates[1], 1)
    e0, heff, info = TensorNetworks.effective_hamiltonian(mps, hammpo, direction = :left)
    Eanalytic = -(cos(2 * theta)^2 + h * sin(2 * theta) * cos(phi) + g * cos(2 * theta))
    #Empo = expectation_value(mps,hammpo)
    @test E ≈ e0 ≈ Eanalytic

    mps = canonicalize(randomUMPS(ComplexF64, 1, 2, 1))
    E = expectation_value(mps, hamgates[1], 1)
    e0, heff, info = TensorNetworks.effective_hamiltonian(mps, hammpo, direction = :left)
    #Empo = expectation_value(mps,hammpo)
    @test E ≈ e0
end

@testset "iterative_compression" begin
    theta = 2 * pi * rand()
    phi = 2 * pi * rand()
    N = 10
    trunc = TruncationArgs(10, 1e-12, true)
    target = LCROpenMPS([qubit(2 * pi * rand(), 2 * pi * rand()) for k in 1:N], truncation = trunc)
    guess = canonicalize(randomLCROpenMPS(N, 2, 10))

    mps = TensorNetworks.iterative_compression(target, guess)
    @test scalar_product(mps, target) ≈ 1
    @test scalar_product(mps, guess) ≈ scalar_product(target, guess)

    target = canonicalize(randomLCROpenMPS(N, 2, 3)) + canonicalize(randomLCROpenMPS(N, 2, 3))
    mps = TensorNetworks.iterative_compression(target, guess)
    @test scalar_product(mps, target) ≈ norm(target)
    @test scalar_product(mps, guess) ≈ scalar_product(target, guess) / norm(target)
end

using DoubleFloats
@testset "DoubleFloats" begin
    # DMRG
    N = 5
    Dmax = 10
    mps = randomLCROpenMPS(N, 2, Dmax, T = ComplexDF64)
    @test eltype(mps) == GenericSite{ComplexDF64}
    @test norm(mps) ≈ 1
    @test eltype(norm(mps)) == real(ComplexDF64)
    @test eltype(canonicalize(mps)) == GenericSite{ComplexDF64}

    T = ComplexDF64
    mps = randomLCROpenMPS(N, 2, Dmax, T = T)
    ham = IsingMPO(N, 1, T(0), 0)
    states, energies = eigenstates(ham, mps, 5; precision = 1e-20)
    @test sort(energies) ≈ -[N - 1, N - 1, N - 3, N - 3, N - 3]
    @test eltype(energies) == real(T)

    # Iterative compression
    N = 10
    trunc = TruncationArgs(10, 1e-20, true)
    target = LCROpenMPS([qubit(2 * pi * rand(real(T)), 2 * pi * rand(real(T))) for k in 1:N], truncation = trunc)
    guess = canonicalize(randomLCROpenMPS(N, 2, 10, T = T))
    mps = TensorNetworks.iterative_compression(target, guess)
    @test eltype(mps) == GenericSite{T}
    @test scalar_product(mps, target) ≈ 1
    @test scalar_product(mps, guess) ≈ scalar_product(target, guess)

    # UMPS
    theta = 2 * pi * rand(real(T))
    phi = 2 * pi * rand(real(T))
    h = rand(real(T))
    g = rand(real(T))
    mps = TensorNetworks.productUMPS(theta, phi)
    @test eltype(mps[1]) == T
    hammpo = MPO(IsingMPO(5, 1, h, g)[3])
    hamgates = isingHamGates(5, 1, h, g)[2:3]
    E = expectation_value(mps, hamgates[1], 1)
    @test isa(E, T)
    e0, heff, info = TensorNetworks.effective_hamiltonian(mps, hammpo, direction = :left)
    @test isa(e0, T)
    Eanalytic = -(cos(2 * theta)^2 + h * sin(2 * theta) * cos(phi) + g * cos(2 * theta))
    @test E ≈ e0 ≈ Eanalytic

    #TEBD
end

@testset "MPSSum" begin
    mps = canonicalize(randomOpenMPS(5, 2, 5))
    smps = mps + mps
    @test length(mps) == length(smps)
    parityop = MPO(fill(MPOsite(sz), 5))
    parity(state) = expectation_value(state, parityop)
    mpsplus = (TensorNetworks.apply_local_op(mps, Gate(sz)) + mps) / sqrt(2)
    mpsplus = mpsplus / (norm(mpsplus))
    mpsminus = (TensorNetworks.apply_local_op(mps, Gate(sz)) - mps) / sqrt(2)
    mpsminus = mpsminus / (norm(mpsminus))
    @test all(iscanonical.(mpsplus))
    @test all(iscanonical.(mpsminus))
    @test norm(mpsplus) ≈ norm(mpsminus) ≈ 1
    @test abs(scalar_product(mpsplus, mpsminus)) < 1e-10

    denseplus = TensorNetworks.dense(mpsplus)
    denseminus = TensorNetworks.dense(mpsminus)
    @test norm(denseplus) ≈ norm(denseminus) ≈ 1
    @test abs(scalar_product(denseplus, denseminus)) < 1e-10
    @test all(iscanonical.(denseplus))
    @test all(iscanonical.(denseminus))
    denseplus = canonicalize(denseplus)
    denseminus = canonicalize(denseminus)
    @test all(iscanonical.(denseplus))
    @test all(iscanonical.(denseminus))
    @test norm(denseplus) ≈ norm(denseminus) ≈ 1
    @test abs(scalar_product(denseplus, denseminus)) < 1e-10
end

@testset "MPOSum" begin
    N = 10
    mpo = TensorNetworks.IsingMPO(N, 1.0, 1.0, 0.0)
    mposum = mpo + 2 * mpo
    @test typeof(mposum) <: TensorNetworks.MPOSum
    @test typeof(mposum[1]) <: TensorNetworks.MPOSiteSum
    @test eltype(mposum) == typeof((mpo+mpo)[1])
    @test length(mposum) == N
    @test size(mposum[1]) == (2 * size(mpo[1], 1), 2, 2, 2 * size(mpo[1], 4))

    mps = randomLCROpenMPS(N, 2, 5)
    @test 3 * expectation_value(mps, mpo) ≈ expectation_value(mps, mposum)
end

@testset "Misc" begin
    N = 5
    qubits = [qubit(rand(), rand()) for k in 1:N]
    mps = LCROpenMPS(qubits)
    indices = rand([1, 2], N)
    wf = TensorNetworks.evaluate_wavefunction(mps, indices)
    @test wf ≈ prod([q[1, indices[k], 1] for (k, q) in enumerate(qubits)])
    d = 5
    sites = [randomLeftOrthogonalSite(1, d, d), randomLeftOrthogonalSite(d, d, 1)]
    mps = LCROpenMPS(sites)
    for (n1, n2) in Iterators.product(1:d, 1:d)
        wf = TensorNetworks.evaluate_wavefunction(mps, [n1, n2])
        @test wf ≈ transpose(sites[1][1, n1, :]) * sites[2][:, n2, 1]
    end

    d = 2
    sites = [randomLeftOrthogonalSite(1, d, d), randomLeftOrthogonalSite(d, d, 1)]
    mps = LCROpenMPS(sites)
    mposite = MPOsite(si)
    mpo = MPO([mposite, mposite])
    lp = mpo * mps
    @test scalar_product(lp, mps) ≈ scalar_product(mps, mps) ≈ scalar_product(lp, lp)
    @test TensorNetworks.dense(lp[1]) ≈ mps[1]

    mposite = MPOsite(sz)
    mpo = MPO([mposite, mposite])
    lp = mpo * mps
    @test scalar_product(lp, mps) ≈ TensorNetworks.matrix_element(mps, mpo, mps)
    @test TensorNetworks.dense(lp[1]) ≈ TensorNetworks.dense(mposite * mps[1])
end

@testset "LazyProduct" begin
    site = randomGenericSite(5, 2, 5, ComplexF64)
    mposite = MPOsite(sz)
    lp = mposite * site
    sites = (mposite, site)
    @test typeof(lp) == TensorNetworks.LazySiteProduct{ComplexF64,typeof(sites),3}
    @test lp.sites == sites
    lp2 = mposite * lp
    @test typeof(lp2) == TensorNetworks.LazySiteProduct{ComplexF64,typeof((mposite, sites...)),3}
    @test lp2.sites == (mposite, sites...)

    lpd = TensorNetworks.dense(lp)
    @test lpd == TensorNetworks.multiply(mposite, site)
    lpd2 = TensorNetworks.dense(lp2)
    @test lpd2 == TensorNetworks.multiply(mposite, lpd)

    mps = randomLCROpenMPS(5,2,5);
    mpo = IsingMPO(5,1,1,1.0)
    lp = mpo*mps
    @test typeof(lp) <: TensorNetworks.LazyProduct
    @test lp.mpos == (mpo,)
    @test lp.mps == mps
    @test typeof(lp[1]) <: TensorNetworks.LazySiteProduct
end