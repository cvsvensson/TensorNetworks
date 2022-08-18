#TODO compress MPO https://arxiv.org/pdf/1611.02498.pdf
abstract type AbstractMPOSite{T} <: AbstractArray{T,4} end

struct MPOSite{T,S<:AbstractArray{T,4}} <: AbstractMPOSite{T}
    data::S
end
const DenseMPOSite{T} = MPOSite{T,Array{T,4}}
Base.getindex(g::MPOSite, I::Vararg{Int,4}) = getindex(data(g), I...)
operatorlength(::AbstractMPOSite) = 1

MPOSite{K}(site::MPOSite, dir) where {K} = MPOSite{K}(data(site))

function MPOSite(op::Array{<:Number,2})
    sop = size(op)
    return MPOSite(reshape(op, 1, sop[1], sop[2], 1))
end

# function MPOsite(op::Array{<:Number,4})
#     return MPOsite(op, norm(op - conj(permutedims(op,[1,3,2,4]))) ≈ 0)

data(site::MPOSite) = site.data
Base.eltype(::MPOSite{T}) where {T} = T
# LinearAlgebra.ishermitian(site::MPOsite) = site.ishermitian
# isunitary(site::MPOsite) = site.isunitary
reverse_direction(site::MPOSite) = MPOSite(permutedims(data(site), [4, 2, 3, 1]))
Base.transpose(site::MPOSite) = MPOSite(permutedims(data(site), [1, 3, 2, 4]))
Base.adjoint(site::MPOSite) = MPOSite(conj(permutedims(data(site), [1, 3, 2, 4])))
Base.conj(site::MPOSite) = MPOSite(conj(data(site)))
Base.permutedims(site::MPOSite, perm) = MPOSite(permutedims(site.data, perm))

Base.:+(site1::MPOSite, site2::MPOSite) = MPOSite(data(site1) .+ data(site2))

abstract type AbstractMPO{T<:Number,MPOSite} <: AbstractVector{MPOSite} end

struct MPO{T<:Number,S} <: AbstractMPO{T,S}
    data::Vector{S}
    function MPO(sites::Vector{S}) where S
        new{eltype(S),S}(sites)
    end
end

sites(mpo::MPO) = mpo.data
Base.IndexStyle(::Type{<:AbstractMPO}) = IndexLinear()
Base.size(mpo::AbstractMPO) = size(sites(mpo))
operatorlength(mpo::AbstractMPO) = length(mpo)

struct ScaledIdentityMPOSite{T} <: AbstractMPOSite{T}
    data::T
    function ScaledIdentityMPOSite(scaling::T) where T
        new{T}(scaling)
    end
end
data(site::ScaledIdentityMPOSite) = site.data
const IdentityMPOSite = ScaledIdentityMPOSite(true)

# MPOsite(s::ScaledIdentityMPOSite) = s
# MPOsite{K}(s::ScaledIdentityMPOSite) where {K} = ScaledIdentityMPOSite{K}(data(s))
#Base.length(mpo::ScaledIdentityMPOSite) = 1

function Base.size(::ScaledIdentityMPOSite, i::Integer)
    if i == 1 || i == 4
        return 1
    else
        @error "Physical dimension of ScaledIdentityMPOSite is arbitrary"
    end
end

LinearAlgebra.ishermitian(mpo::ScaledIdentityMPOSite) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPOSite) = data(mpo)' * data(mpo) ≈ 1
Base.:*(x::K, g::ScaledIdentityMPOSite) where {K<:Number} = ScaledIdentityMPOSite(x * data(g))
Base.:*(g::ScaledIdentityMPOSite, x::K) where {K<:Number} = ScaledIdentityMPOSite(x * data(g))
Base.:/(g::ScaledIdentityMPOSite, x::K) where {K<:Number} = inv(x) * g
auxillerate(mpo::ScaledIdentityMPOSite) = mpo
Base.show(io::IO, g::ScaledIdentityMPOSite) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPOSite"))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPOSite) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPOSite"))

reverse_direction(site::ScaledIdentityMPOSite) = site
Base.transpose(site::ScaledIdentityMPOSite) = site
Base.adjoint(site::ScaledIdentityMPOSite) = conj(site)
Base.conj(site::ScaledIdentityMPOSite) = ScaledIdentityMPOSite(conj(data(site)))
Base.:+(site1::ScaledIdentityMPOSite, site2::ScaledIdentityMPOSite) = ScaledIdentityMPOSite(data(site1) + data(site2))

Base.:(==)(mpo1::ScaledIdentityMPOSite, mpo2::ScaledIdentityMPOSite) = data(mpo1) == data(mpo2)

function Base.getindex(g::ScaledIdentityMPOSite{T}, I::Vararg{Int,4}) where T
    @assert I[1] == I[4] == 1 "Accessing out of bounds index on ScaledIdentityMPOSite "
    val = I[2] == I[3] ? one(T) : zero(T)
    return data(g) * val
end

struct ScaledIdentityMPO{T} <: AbstractMPO{T,ScaledIdentityMPOSite{T}}
    data::T
    length::Int
    function ScaledIdentityMPO(scaling::T, n::Integer) where {T}
        new{T}(scaling, n)
    end
end
Base.IndexStyle(::Type{<:ScaledIdentityMPO}) = IndexLinear()
Base.getindex(g::ScaledIdentityMPO, i::Integer) = g.data^(1 / length(g)) * IdentityMPOSite
Base.getindex(g::ScaledIdentityMPO, I) = g.data^(1 / length(g)) * fill(IdentityMPOSite, length(I))
data(g::ScaledIdentityMPO) = g.data
sites(g::ScaledIdentityMPO) = g[1:length(g)]


IdentityMPO(n) = ScaledIdentityMPO(true, n)
Base.length(mpo::ScaledIdentityMPO) = mpo.length
operatorlength(mpo::ScaledIdentityMPO) = mpo.length
LinearAlgebra.ishermitian(mpo::ScaledIdentityMPO) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPO) = data(mpo)' * data(mpo) ≈ 1
Base.:*(x::K, g::ScaledIdentityMPO) where {K<:Number} = ScaledIdentityMPO(x * data(g), length(g))
Base.:*(g::ScaledIdentityMPO, x::K) where {K<:Number} = ScaledIdentityMPO(x * data(g), length(g))
Base.:/(g::ScaledIdentityMPO, x::K) where {K<:Number} = inv(x) * g
Base.show(io::IO, g::ScaledIdentityMPO) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPO of length ", length(g)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPO) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPO of length ", length(g)))

Base.:(==)(mpo1::ScaledIdentityMPO, mpo2::ScaledIdentityMPO) = data(mpo1) == data(mpo2) && length(mpo1) == length(mpo2)
# struct HermitianMPO{T<:Number} <: AbstractMPO{T}
#     data::MPO{T}
#     function HermitianMPO(mpo::MPO{T}) where {T}
#         mpo = MPO([(m+ conj(permutedims(m,[1,3,2,4])))/2 for m in mpo.data])
#         new{T}(mpo)
#     end
#     function HermitianMPO(sites::Vector{MPOsite{T}}) where {T}
#         mpo = MPO([(site + conj(permutedims(site,[1,3,2,4])))/2 for site in sites])
#         new{T}(mpo)
#     end
# end

MPO(mpo::MPOSite) = MPO([mpo])
MPO(op::Array{<:Number,2}) = MPO(MPOSite(op))
MPO(ops::Vector{Matrix{T}}) where {T} = MPO(map(MPOSite, ops))
MPO(ops::Vector{Array{T,4}}) where {T} = MPO(map(MPOSite, ops))
MPO(ops::Vector{<:ScaledIdentityMPOSite}) = prod(data.(ops))*IdentityMPO(length(ops))
data(mpo::MPO) = mpo.data
# HermitianMPO(mpo::MPOsite) = HermitianMPO(MPO([mpo]))
# HermitianMPO(op::Array{T,2}) where {T<:Number} = HermitianMPO(MPOsite(op))
# HermitianMPO(ops::Array{Array{T,4},1}) where {T<:Number} = HermitianMPO(map(op->MPOsite(op),ops))

Base.size(mposite::MPOSite) = size(data(mposite))
# Base.length(mpo::MPOsite) = 1
# Base.length(mpo::MPO) = length(data(mpo))
# Base.IndexStyle(::Type{<:MPOsite}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(mpo::MPOSite, i::Integer) = mpo.data[i]
Base.@propagate_inbounds Base.getindex(mpo::AbstractMPO, i::Integer) = mpo.data[i]
#Base.setindex!(mpo::MPOsite, v, I::Vararg{Integer,4}) = (mpo.data[I] = v)
#Base.setindex!(mpo::AbstractMPO, v, I::Vararg{Integer,N}) where {N} = (mpo.data[I] = v)


"""
	auxillerate(mpo)

Return tensor⨂Id_aux
"""
function auxillerate(mpo::MPOSite)
    sop = size(mpo)
    d = sop[2]
    idop = Matrix{eltype(mpo)}(I, d, d)
    @tensor tens[:] := idop[-3, -5] * mpo.data[-1, -2, -4, -6]
    return MPOSite(reshape(tens, sop[1], d^2, d^2, sop[4]))
end

auxillerate(mpo::MPO) = MPO(auxillerate.(mpo.data))
#auxillerate(mpo::HermitianMPO) = HermitianMPO(auxillerate.(mpo.data))

# %% Todo
"""
gives the mpo corresponding to a*mpo1 + b*mpo2.
"""
function addmpos(mpo1, mpo2, a, b, Dmax, tol = 0) #FIXME
    L = length(mpo1)
    d = size(mpo1[1])[2]
    mpo = Array{Array{Complex{Float64}}}(L)
    mpo[1] = permutedims(cat(1, permutedims(a * mpo1[1], [4, 1, 2, 3]), permutedims(b * mpo2[1], [4, 1, 2, 3])), [2, 3, 4, 1])
    for i = 2:L-1
        mpo[i] = permutedims([permutedims(mpo1[i], [1, 4, 2, 3]) zeros(size(mpo1[i])[1], size(mpo2[i])[4], d, d); zeros(size(mpo2[i])[1], size(mpo1[i])[4], d, d) permutedims(mpo2[i], [1, 4, 2, 3])], [1, 3, 4, 2])
        if tol > 0 || size(mpo1[i])[3] + size(mpo2[i])[3] > Dmax
            @tensor tmp[-1, -2, -3, -4, -5, -6] := mpo[i-1][-1, -2, -3, 1] * mpo[i][1, -4, -5, -6]
            tmp = reshape(tmp, size(mpo[i-1])[1] * d * d, d * d * size(mpo[i])[4])
            F = svd(tmp)
            U, S, V = truncate_svd(F, Dmax, tol)
            mpo[i-1] = reshape(1 / 2 * U * Diagonal(S), size(mpo[i-1])[1], d, d, D)
            mpo[i] = reshape(2 * V, D, d, d, size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1, permutedims(mpo1[L], [1, 4, 2, 3]), permutedims(mpo2[L], [1, 4, 2, 3])), [1, 3, 4, 2])
    if tol > 0
        @tensor tmp[-1, -2, -3, -4, -5, -6] := mpo[L-1][-1, -2, -3, 1] * mpo[L][1, -4, -5, -6]
        tmp = reshape(tmp, size(mpo[L-1])[1] * d * d, d * d * size(mpo[L])[4])
        F = svd(tmp)
        U, S, V = truncate_svd(F, D, tol)
        mpo[L-1] = reshape(1 / 2 * U * Diagonal(S), size(mpo[L-1])[1], d, d, D)
        mpo[L] = reshape(2 * V, D, d, d, size(mpo[L])[4])
    end
    return mpo
end



function multiplyMPOsites(site1, site2)
    s1 = size(site1)
    s2 = size(site2)
    @tensor temp[:] := data(site1)[-1, -3, 1, -5] * data(site2)[-2, 1, -4, -6]
    return MPOSite(reshape(temp, s1[1] * s2[1], s1[2], s2[3], s1[4] * s2[4]))
end
"""
```multiplyMPOs(mpo1,mpo2)```
"""
function multiplyMPOs(mpo1, mpo2)
    # L = length(mpo1)
    # mpo = similar(mpo1)
    return MPO([multiplyMPOsites(s1, s2) for (s1, s2) in zip(mpo1, mpo2)])
    #     # if c
    #     # @tullio temp[l1,l2,u,d,r1,r2] := data(mpo1[j])[l1,u,c,r1] * data(mpo2[j])[l2,c,d,r2]
    #         #@tensor temp[:] := mpo1[j].data[-1,-3,1,-5] * conj(mpo2[j].data[-2,-4,1,-6])
    #     # else
    #     @tensor temp[:] := data(mpo1[j])[-1,-3,1,-5] * data(mpo2[j])[-2,1,-4,-6]
    #     # end
    #     s=size(temp)
    #     mpo[j] = MPOsite(reshape(temp,s[1]*s[2],s[3],s[4],s[5]*s[6]))
    # end
    # return MPO(mpo)
end


function Matrix(mpo::MPO)
    n = length(mpo)
    T = eltype(mpo[1])
    vl = boundary(OpenBoundary(),mpo,:left)
    tens = SparseArray(reshape(vl,1,1,length(vl)))
    for site in mpo[1:n]
        dat = SparseArray(data(site))
        @tensor tens[out, newout, in, newin, right] := tens[out, in, c] * dat[c, newout, newin, right]
        st = size(tens)
        tens = SparseArray(reshape(tens, st[1] * st[2], st[3] * st[4], st[5]))
    end
    vr = SparseArray(boundary(OpenBoundary(),mpo,:right))
    return @tensor out[:] := tens[-1, -2, 1] * vr[1]
end