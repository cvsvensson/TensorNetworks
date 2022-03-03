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
    # mpo = Array{Array{Complex{Float32},4}}(L)
    mpo = Array{Any}(L)
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
    mpo = Array{Any}(L)
    mpo[1] = Array{ComplexF64}(1, 2, 2, 2)
    mpo[1][1, :, :, :] = reshape([si M], 2, 2, 2)
    mpo[L] = Array{ComplexF64}(2, 2, 2, 1)
    mpo[L][:, :, :, 1] = permutedims(reshape([M si], 2, 2, 2), [3, 1, 2])
    for i = 2:L-1
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
function IsingMPO(L, J, h, g, shift = 0)
    T = promote_type(eltype.((J, h, g))...)
    mpo = Vector{Array{T,4}}(undef, L)
    mpo[1] = zeros(T, 1, 2, 2, 3)
    mpo[1][1, :, :, :] = reshape([si -J * sz -h * sx - g * sz + shift * si / L], 2, 2, 3)
    mpo[L] = zeros(T, 3, 2, 2, 1)
    mpo[L][:, :, :, 1] = permutedims(reshape([-h * sx - g * sz + shift * si / L sz si], 2, 2, 3), [3, 1, 2])
    for i = 2:L-1
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
# function KitaevMPO3(N, t, Δ, U, μ)
#     T = complex(promote_type(eltype.(((t+Δ)/2,(Δ-t)/2,U,μ/2))...))
#     mpo = Vector{Array{T,4}}(undef,N)
#     D = 5
#     firstrow = [si  -sx*(t+Δ)/2  sy*(Δ-t)/2  U*sz  -μ/2*sz]
#     mpo[1] = zeros(T,1,2,2,D)
#     mpo[1][1,:,:,:] = reshape(firstrow,2,2,D)
#     for i=2:N-1
#         help = zeros(T,D,2,2,D)
#         help[1,:,:,:] = firstrow
#         help[2,:,:,D] = sx
#         help[3,:,:,D] = sy
#         help[4,:,:,D] = sz
#         help[D,:,:,D] = si
#         # help[9,:,:,9] = si
#         mpo[i] = help
#     end
#     mpo[N] = zeros(T,D,2,2,1)
#     mpo[N][1,:,:,1] = -μ/2 *sz
#     mpo[N][2,:,:,1] = sx
#     mpo[N][3,:,:,1] = sy
#     mpo[N][4,:,:,1] = sz
#     mpo[N][D,:,:,1] = si

#     return MPO(mpo)
# end

function KitaevMPO(N, t, Δ, U, μ; type = ComplexF64)
    center = _KitaevMPO_center(t, Δ, U, μ, type = type)
    mpo = fill(center, N)
    mpo[1] = _KitaevMPO_left(t, Δ, U, μ, type = type)
    mpo[N] = _KitaevMPO_right(t, Δ, U, μ, type = type)
    return MPO(mpo)
end

function KitaevMPO(t, Δ, U, μs::Array; type = ComplexF64)
    N = length(μs)
    mpo = [_KitaevMPO_center(t, Δ, U, μ, type = type) for μ in μs]
    mpo[1] = _KitaevMPO_left(t, Δ, U, μs[1], type = type)
    mpo[N] = _KitaevMPO_right(t, Δ, U, μs[N], type = type)
    return MPO(mpo)
end

function DisorderedKitaevMPO(N, t, Δ, U, μ::Tuple{K,T}; type = ComplexF64) where {K,T}
    (mμ, sμ) = μ
    μs = sμ * randn(N) .+ mμ
    mpo = [_KitaevMPO_center(t, Δ, U, μ, type = type) for μ in μs]
    mpo[1] = _KitaevMPO_left(t, Δ, U, μs[1], type = type)
    mpo[N] = _KitaevMPO_right(t, Δ, U, μs[N], type = type)
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

function _KitaevMPO_center(t, Δ, U, μ; type = ComplexF64)
    D = 5
    mposite = zeros(type, D, 2, 2, D)
    mposite[1, :, :, :] = [si -sx * (t + Δ) / 2 sy * (Δ - t) / 2 U * sz -μ / 2 * sz]
    mposite[2, :, :, D] = sx
    mposite[3, :, :, D] = sy
    mposite[4, :, :, D] = sz
    mposite[D, :, :, D] = si
    return mposite
end
function _KitaevMPO_left(t, Δ, U, μ; type = ComplexF64)
    D = 5
    mposite = zeros(type, 1, 2, 2, D)
    mposite[1, :, :, :] = reshape([si -sx * (t + Δ) / 2 sy * (Δ - t) / 2 U * sz -μ / 2 * sz], 2, 2, D)
    return mposite
end
function _KitaevMPO_right(t, Δ, U, μ; type = ComplexF64)
    D = 5
    mposite = zeros(type, D, 2, 2, 1)
    mposite[1, :, :, 1] = -μ / 2 * sz
    mposite[2, :, :, 1] = sx
    mposite[3, :, :, 1] = sy
    mposite[4, :, :, 1] = sz
    mposite[D, :, :, 1] = si
    return mposite
end
"""
    HeisenbergMPO(lattice sites,Jx,Jy,Jz,transverse)

Returns the Heisenberg hamiltonian as an MPO
"""
# function HeisenbergMPO(L, Jx, Jy, Jz, h)
#     mpo = Vector{Array{ComplexF64,4}}(undef,L)
#     mpo[1] = zeros(ComplexF64,1,2,2,5)
#     mpo[1][1,:,:,:] = reshape([si Jx*sx Jy*sy Jz*sz h*sx], 2,2,5)
#     mpo[L] =zeros(ComplexF64,5,2,2,1)
#     mpo[L][:,:,:,1] = permutedims(reshape([h*sx sx sy sz si], 2,2,5), [3,1,2])

#     for i=2:L-1
#         # hardcoded implementation of index structure (a,i,j,b):
#         help = zeros(ComplexF64,5,2,2,5)
#         help[1,:,:,1] = help[5,:,:,5] = si
#         help[1,:,:,2] = Jx*sx
#         help[1,:,:,3] = Jy*sy
#         help[1,:,:,4] = Jz*sz
#         help[1,:,:,5] = h*sx
#         help[2,:,:,5] = sx
#         help[3,:,:,5] = sy
#         help[4,:,:,5] = sz
#         #help[2,:,:,1:4] = help[3,:,:,1:4] = help[4,:,:,1:4] = help[5,:,:,1:4] = s0
#         mpo[i] = help
#     end
#     return MPO(mpo)
# end

# function HeisenbergS1MPO(L, Jx, Jy, Jz, h)
#     mpo = Vector{Array{ComplexF64,4}}(undef,L)
#     id = Matrix{ComplexF64}(I,3,3)
#     mpo[1] = zeros(ComplexF64,1,3,3,5)
#     mpo[1][1,:,:,:] = reshape([id Jx*sx1 Jy*sy1 Jz*sz1 h*sx1], 3,3,5)
#     mpo[L] =zeros(ComplexF64,5,3,3,1)
#     mpo[L][:,:,:,1] = permutedims(reshape([h*sx1 sx1 sy1 sz1 id], 3,3,5), [3,1,2])
#     for i=2:L-1
#         # hardcoded implementation of index structure (a,i,j,b):
#         help = zeros(ComplexF64,5,3,3,5)
#         help[1,:,:,1] = help[5,:,:,5] = id
#         help[1,:,:,2] = Jx*sx1
#         help[1,:,:,3] = Jy*sy1
#         help[1,:,:,4] = Jz*sz1
#         help[1,:,:,5] = h*sx1
#         help[2,:,:,5] = sx1
#         help[3,:,:,5] = sy1
#         help[4,:,:,5] = sz1
#         #help[2,:,:,1:4] = help[3,:,:,1:4] = help[4,:,:,1:4] = help[5,:,:,1:4] = s0
#         mpo[i] = help
#     end
#     return MPO(-mpo)
# end

function HeisenbergMPO(S, L, Jx, Jy, Jz, h; type = ComplexF64)
    center = _HeisenbergMPO_center(S, Jx, Jy, Jz, h; type = type)
    left = _HeisenbergMPO_left(S, Jx, Jy, Jz, h; type = type)
    right = _HeisenbergMPO_right(S, Jx, Jy, Jz, h; type = type)
    mpo = Vector{Array{type,4}}(undef, L)
    mpo = fill(center, L)
    mpo[1] = left
    mpo[end] = right
    return MPO(mpo)
end

function _HeisenbergMPO_left(S, Jx, Jy, Jz, h; type = ComplexF64)
    s2 = Int(2S)
    L = zeros(type, 1, s2 + 1, s2 + 1, 5)
    L[1, :, :, :] = reshape([Matrix{type}(I, s2 + 1, s2 + 1) Jx * Sx(S) Jy * Sy(S) Jz * Sz(S) h * Sx(S)], s2 + 1, s2 + 1, 5)
    return L
end
function _HeisenbergMPO_right(S, Jx, Jy, Jz, h; type = ComplexF64)
    s2 = Int(2S)
    R = zeros(type, 5, s2 + 1, s2 + 1, 1)
    R[:, :, :, 1] = permutedims(reshape([-h * Sx(S) Sx(S) Sy(S) Sz(S) Matrix{type}(I, s2 + 1, s2 + 1)], s2 + 1, s2 + 1, 5), [3, 1, 2])
    return R
end

function _HeisenbergMPO_center(S, Jx, Jy, Jz, h; type = ComplexF64)
    mposite = zeros(type, 5, 2S + 1, 2S + 1, 5)
    mposite[1, :, :, 1] = mposite[5, :, :, 5] = Matrix{type}(I, 2S + 1, 2S + 1)
    mposite[1, :, :, 2] = Jx * Sx(S)
    mposite[1, :, :, 3] = Jy * Sy(S)
    mposite[1, :, :, 4] = Jz * Sz(S)
    mposite[1, :, :, 5] = h * Sx(S)
    mposite[2, :, :, 5] = Sx(S)
    mposite[3, :, :, 5] = Sy(S)
    mposite[4, :, :, 5] = Sz(S)
    return mposite
end

"""
    TwoSiteHamToMPO(ham,L)

Returns the MPO for a 2-site Hamiltonian
"""
function TwoSiteHamToMPO(ham, L)
    d = size(ham)[1]
    mpo = Array{Any}(L)
    tmp = reshape(permutedims(ham, [1, 3, 2, 4]), d * d, d * d)
    U, S, V = svd(tmp)
    U = reshape(U * Diagonal(sqrt.(S)), d, d, size(S)[1])
    V = reshape(Diagonal(sqrt.(S)) * V', size(S)[1], d, d)
    mpo[1] = permutedims(reshape(U, d, d, size(S), 1), [1, 4, 3, 2])
    mpo[L] = permutedims(reshape(V, size(S), d, d, 1), [2, 1, 4, 3])
    @tensor begin
        tmpEven[-1, -2, -3, -4] := V[-2, -1, 1] * U[1, -4, -3]
        tmpOdd[-1, -2, -3, -4] := U[-1, 1, -3] * V[-2, 1, -4]
    end
    for i = 2:L-1
        if iseven(i)
            mpo[i] = tmpEven
        else
            mpo[i] = tmpOdd
        end
    end
    return mpo
end
