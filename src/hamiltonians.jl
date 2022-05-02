"""
    isingHamBlocks(L,J,h,g)

Return the Ising hamiltonian as a list of matrices
"""
function isingHamBlocks(L, J, h, g)
    T = promote_type(eltype.((J, h, g))...)
    blocks = Vector{Array{T,2}}(undef, L - 1)
    for i = 1:L-1
        if i == 1
            blocks[i] = -(J * ZZ + h / 2 * (2XI + IX) + g / 2 * (2 * ZI + IZ))
        elseif i == L - 1
            blocks[i] = -(J * ZZ + h / 2 * (XI + 2IX) + g / 2 * (ZI + 2 * IZ))
        else
            blocks[i] = -(J * ZZ + h / 2 * (XI + IX) + g / 2 * (ZI + IZ))
        end
    end
    return blocks
end

"""
    isingHamGates(L,J,h,g)

Return the Ising hamiltonian as a list of 2-site gates
"""
function isingHamGates(L, J, h, g)
    T = promote_type(eltype.((J, h / 2, g / 2))...)
    gates = Vector{GenericSquareGate{T,4}}(undef, L - 1)
    for i = 1:L-1
        if i == 1
            gate = reshape(-(J * ZZ + h / 2 * (2XI + IX) + g / 2 * (2 * ZI + IZ)), 2, 2, 2, 2)
        elseif i == L - 1
            gate = reshape(-(J * ZZ + h / 2 * (XI + 2IX) + g / 2 * (ZI + 2 * IZ)), 2, 2, 2, 2)
        else
            gate = reshape(-(J * ZZ + h / 2 * (XI + IX) + g / 2 * (ZI + IZ)), 2, 2, 2, 2)
        end
        gates[i] = GenericSquareGate(gate)
    end
    return gates
end


"""
    IdentityMPO(lattice sites, phys dims)

Return the identity MPO
"""
function IdentityMPO(L, d) #FIXME Type stability. Maybe introduce a new type IdentityMPO?
    mpo = Array{Array{ComplexF64,4}}(undef,L)
    #mpo = Array{Any}(L)
    for i = 1:L
        mpo[i] = Array{ComplexF64}(1, d, d, 1)
        mpo[i][1, :, :, 1] = eye(d)
    end
    return MPO(mpo)
end

"""
    translationMPO(lattice sites, matrix)

Returns a translationally invariant one-site mpo
"""
function translationMPO(L, M)
    mpo = Array{Array{ComplexF64,4}}(undef, L)
    for i = 1:L
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex128}(2, 2, 2, 2)
        help[1, :, :, 1] = help[2, :, :, 2] = si
        help[1, :, :, 2] = M
        help[2, :, :, 1] = s0
        mpo[i] = help
    end

    return MPO(mpo)
end


"""
IsingMPO(lattice sites, J, transverse, longitudinal[, shift=0])

Returns the Ising hamiltonian as an MPO
"""
function IsingMPO(L, J, h, g, shift=0)
    T = promote_type(eltype.((J, h, g))...)
    mpo = Vector{Array{T,4}}(undef, L)
    # mpo[1] = zeros(T, 1, 2, 2, 3)
    # mpo[1][1, :, :, :] = reshape([si -J * sz -h * sx - g * sz + shift * si / L], 2, 2, 3)
    # mpo[L] = zeros(T, 3, 2, 2, 1)
    # mpo[L][:, :, :, 1] = permutedims(reshape([-h * sx - g * sz + shift * si / L sz si], 2, 2, 3), [3, 1, 2])
    for i = 1:L
        # hardcoded implementation of index structure (a,i,j,b):
        help = zeros(T, 3, 2, 2, 3)
        help[1, :, :, 1] = help[3, :, :, 3] = si #Maybe this should not be here at i=L-1?
        help[1, :, :, 2] = -J * sz
        help[1, :, :, 3] = -h * sx - g * sz + shift * si / L
        help[2, :, :, 1] = help[2, :, :, 2] = help[3, :, :, 1] = help[3, :, :, 2] = s0
        help[2, :, :, 3] = sz
        mpo[i] = help
    end
    return MPO(mpo)
end


"""
KitaevMPO(N, t, Δ, U, μ)

Returns the Kitaev spin chain
"""
function KitaevMPO(N, t, Δ, U, μ; type=Float64)
    center = _KitaevMPO_center(t, Δ, U, μ, type=type)
    mpo = fill(center, N)
    return MPO(mpo)
end

function KitaevMPO(t, Δ, U, μs::Array; type=Float64)
    N = length(μs)
    mpo = [_KitaevMPO_center(t, Δ, U, μ, type=type) for μ in μs]
    return MPO(mpo)
end

function DisorderedKitaevMPO(N, t, Δ, U, μ::Tuple{K,T}; type=Float64) where {K,T}
    (mμ, sμ) = μ
    μs = sμ * randn(N) .+ mμ
    mpo = [_KitaevMPO_center(t, Δ, U, μ, type=type) for μ in μs]
    return MPO(mpo)
end

function KitaevGates(t, Δ, U, μs::Array)
    N = length(μs)
    gates = [_KitaevGate_center(t, Δ, U, μ) for μ in μs[1:N-1]]
    gates[1] = _KitaevGate_left(t, Δ, U, μs[1])
    gates[end] = _KitaevGate_right(t, Δ, U, μs[N-1], μs[N])
    return gates
end
_KitaevGate_center(t, Δ, U, μ) = -gXX * (t + Δ) / 2 + gYY * (Δ - t) / 2 + U * gZZ - μ / 2 * gZI
_KitaevGate_left(t, Δ, U, μ) = -gXX * (t + Δ) / 2 + gYY * (Δ - t) / 2 + U * gZZ - μ / 2 * gZI
_KitaevGate_right(t, Δ, U, μ1, μ2) = -gXX * (t + Δ) / 2 + gYY * (Δ - t) / 2 + U * gZZ - μ1 / 2 * gZI - μ2 / 2 * gIZ

function _KitaevMPO_center(t, Δ, U, μ; type=Float64)
    D = 5
    mposite = zeros(type, D, 2, 2, D)
    mposite[1, :, :, :] = [si -sx * (t + Δ) / 2 sy*(1im) * (Δ - t) / 2 U * sz -μ / 2 * sz]
    mposite[2, :, :, D] = sx
    mposite[3, :, :, D] = sy*(-1im)
    mposite[4, :, :, D] = sz
    mposite[D, :, :, D] = si
    return mposite
end

"""
    HeisenbergMPO(lattice sites,Jx,Jy,Jz,transverse)

Returns the Heisenberg hamiltonian as an MPO
"""
function HeisenbergMPO(S, L, Jx, Jy, Jz, h; type=Float64)
    center = _HeisenbergMPO_center(S, Jx, Jy, Jz, h; type=type)
    mpo = Vector{Array{type,4}}(undef, L)
    mpo = fill(center, L)
    return MPO(mpo)
end

function _HeisenbergMPO_center(S, Jx, Jy, Jz, h; type=Float64)
    mposite = zeros(type, 5, 2S + 1, 2S + 1, 5)
    mposite[1, :, :, 1] = mposite[5, :, :, 5] = Matrix{type}(I, 2S + 1, 2S + 1)
    mposite[1, :, :, 2] = Jx * Sx(S)
    mposite[1, :, :, 3] = Jy * Sy(S)*(1im)
    mposite[1, :, :, 4] = Jz * Sz(S)
    mposite[1, :, :, 5] = h * Sx(S)
    mposite[2, :, :, 5] = Sx(S)
    mposite[3, :, :, 5] = Sy(S)*(-1im)
    mposite[4, :, :, 5] = Sz(S)
    return mposite
end

function BD1MPO(N,μ,h, t, α, Δ, Δ1, U, V; type=Float64)
    center = _BD1MPO_center(μ,h, t, α, Δ, Δ1, U, V; type=type)
    #mpo = Vector{Array{type,4}}(undef, N)
    mpo = fill(center, N)
    return MPO(mpo)
end
function _BD1MPO_center(μ,h, t, α, Δ, Δ1, U, V; type=Float64)
    D = 12
    d = 4
    mposite = zeros(type, D, d, d, D)
    mposite[1, :, :, 1] = mposite[D, :, :, D] = Matrix{type}(I, d, d)
    mposite[1, :, :, D] = (-μ - h) / 2 * (ZI+II) + (-μ + h) / 2 * (IZ+II) + Δ/2 * (-XX + YY) + U / 4 * (II + ZI + IZ + ZZ) #(SmSm * ZI - SpSp * ZI)

    #Kinetic terms
    dk = 2
    mposite[1, :, :, dk] = t / 4 * XZ
    mposite[dk, :, :, D] = XI
    mposite[1, :, :, dk+1] = t / 4 * YZ *(1im) #Multiply with i to make everything real
    mposite[dk+1, :, :, D] = YI *(-1im)
    mposite[1, :, :, dk+2] = t / 4 * IX
    mposite[dk+2, :, :, D] = ZX
    mposite[1, :, :, dk+3] = t / 4 * IY *(1im)
    mposite[dk+3, :, :, D] = ZY *(-1im)

    #Spin orbit and Intersite superconductivity
    dsoc = dk + 4
    mposite[1, :, :, dsoc] = -(Δ1/2 + α/4) * XZ
    mposite[dsoc, :, :, D] = ZX
    mposite[1, :, :, dsoc+1] = -(-Δ1/2 + α/4) * YZ *(1im)
    mposite[dsoc+1, :, :, D] = ZY *(-1im)
    mposite[1, :, :, dsoc+2] = -(-Δ1/2 - α/4) * IX
    mposite[dsoc+2, :, :, D] = XI
    mposite[1, :, :, dsoc+3] = -(Δ1/2 - α/4) * IY *(1im)
    mposite[dsoc+3, :, :, D] = YI *(-1im)

    #Intersite interaction
    dinter = dsoc + 4
    mposite[1, :, :, dinter] = V / 4 * (II+ZI)
    mposite[dinter, :, :, D] = (II+ZI)
    mposite[1, :, :, dinter+1] = V / 4 * (II+IZ)
    mposite[dinter+1, :, :, D] = (II+IZ)

    return mposite
end