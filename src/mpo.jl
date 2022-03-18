#TODO compress MPO https://arxiv.org/pdf/1611.02498.pdf
abstract type AbstractMPOsite{T} <: AbstractArray{T,4} end

struct MPOsite{T} <: AbstractMPOsite{T}
    data::Array{T,4}
    # ishermitian::Bool
    # isunitary::Bool
end
Base.getindex(g::MPOsite, I::Vararg{Int,4}) = getindex(data(g), I...)
operatorlength(::AbstractMPOsite) = 1
Base.similar(g::MPOsite) = MPOsite(similar(g.data))

MPOsite{K}(site::MPOsite, dir) where {K} = MPOsite{K}(data(site))

function MPOsite(op::Array{<:Number,2})
    sop = size(op)
    return MPOsite(reshape(op, 1, sop[1], sop[2], 1))
end

# function MPOsite(op::Array{<:Number,4})
#     return MPOsite(op, norm(op - conj(permutedims(op,[1,3,2,4]))) ≈ 0)

data(site::MPOsite) = site.data
Base.eltype(::MPOsite{T}) where {T} = T
# LinearAlgebra.ishermitian(site::MPOsite) = site.ishermitian
# isunitary(site::MPOsite) = site.isunitary
reverse_direction(site::MPOsite) = MPOsite(permutedims(data(site), [4, 2, 3, 1]))
Base.transpose(site::MPOsite) = MPOsite(permutedims(data(site), [1, 3, 2, 4]))
Base.adjoint(site::MPOsite) = MPOsite(conj(permutedims(data(site), [1, 3, 2, 4])))
Base.conj(site::MPOsite) = MPOsite(conj(data(site)))
Base.permutedims(site::MPOsite, perm) = MPOsite(permutedims(site.data, perm))

#Base.:+(site1::MPOsite, site2::MPOsite) = MPOsite(data(site1) .+ data(site2))
Base.:*(x::K, site::MPOsite) where {K<:Number} = MPOsite(x * data(site))
Base.:*(site::MPOsite, x::K) where {K<:Number} = x * site
Base.:/(site::MPOsite, x::K) where {K<:Number} = inv(x) * site
Base.copy(site::MPOsite) = MPOsite(copy(data(site)))

link(site::MPOsite, dir) = I

abstract type AbstractMPO{T<:Number} <: AbstractVector{MPOsite{T}} end

struct MPO{S,T<:Number} <: AbstractMPO{T}
    data::Vector{S}
    boundary::Vector{T}
    function MPO(sites::Vector{S}, boundary::Vector{T}) where {S<:AbstractMPOsite,T}
        K = promote_type(eltype.(sites)..., T)
        new{S,K}(sites, boundary)
    end
end
function MPO(ss::Vector{S}) where {S<:AbstractMPOsite}
    T = promote_type(eltype.(ss)...)
    MPO(ss, fill(one(T), length(sites(ss[1]))))
end
Base.similar(mpo::MPO) = MPO(similar.(sites(mpo)), similar(mpo.boundary))
sites(mpo::MPO) = mpo.data
Base.IndexStyle(::Type{<:AbstractMPO}) = IndexLinear()
Base.size(mpo::AbstractMPO) = size(sites(mpo))
operatorlength(mpo::AbstractMPO) = length(mpo)

function Base.:*(x::K, mpo::MPO) where {K<:Number}
    mpo2 = copy(mpo)
    mpo2[1] = x * mpo[1]
    return mpo2
end
Base.:*(mpo::MPO, x::K) where {K<:Number} = x * mpo
Base.:/(mpo::MPO, x::K) where {K<:Number} = inv(x) * mpo
Base.copy(mpo::MPO) = MPO(copy.(mpo), copy(mpo.boundary))

Base.setindex!(mpo::MPO, v, i::Integer) = mpo.data[i] = v

struct ScaledIdentityMPOsite{T} <: AbstractMPOsite{T}
    data::T
    dim::Int
    function ScaledIdentityMPOsite(scaling::T, dim) where {T<:Number}
        new{T}(scaling, dim)
    end
end
data(site::ScaledIdentityMPOsite) = site.data
IdentityMPOsite(d) = ScaledIdentityMPOsite(true, d)

# MPOsite(s::ScaledIdentityMPOsite) = s
# MPOsite{K}(s::ScaledIdentityMPOsite) where {K} = ScaledIdentityMPOsite{K}(data(s))
#Base.length(mpo::ScaledIdentityMPOsite) = 1
Base.size(mpo::ScaledIdentityMPOsite) = (1, mpo.dim, mpo.dim, 1)
function Base.size(mpo::ScaledIdentityMPOsite, i::Integer)
    if i == 1 || i == 4
        return 1
    else
        return mpo.dim
    end
end

LinearAlgebra.ishermitian(mpo::ScaledIdentityMPOsite) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPOsite) = data(mpo)' * data(mpo) ≈ 1
Base.:*(x::K, g::ScaledIdentityMPOsite) where {K<:Number} = ScaledIdentityMPOsite(x * data(g), g.dim)
Base.:*(g::ScaledIdentityMPOsite, x::K) where {K<:Number} = x * g
Base.:/(g::ScaledIdentityMPOsite, x::K) where {K<:Number} = inv(x) * g
auxillerate(mpo::ScaledIdentityMPOsite) = mpo
Base.show(io::IO, g::ScaledIdentityMPOsite) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPOsite"))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPOsite) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPOsite"))

reverse_direction(site::ScaledIdentityMPOsite) = site
Base.transpose(site::ScaledIdentityMPOsite) = site
Base.adjoint(site::ScaledIdentityMPOsite) = conj(site)
Base.conj(site::ScaledIdentityMPOsite) = ScaledIdentityMPOsite(conj(data(site)), site.dim)
Base.:+(site1::ScaledIdentityMPOsite, site2::ScaledIdentityMPOsite) = ScaledIdentityMPOsite(data(site1) + data(site2), site1.dim)

Base.:(==)(mpo1::ScaledIdentityMPOsite, mpo2::ScaledIdentityMPOsite) = (data(mpo1) == data(mpo2) && mpo1.dim == mpo2.dim)

function Base.getindex(g::ScaledIdentityMPOsite, I::Vararg{Int,4})
    @assert I[1] == I[4] == 1 "Accessing out of bounds index on ScaledIdentityMPOsite "
    val = I[2] == I[3] ? 1 : 0
    return data(g) * val
end

struct ScaledIdentityMPO{T} <: AbstractMPO{T}
    data::T
    length::Int
    dim::Int
    function ScaledIdentityMPO(scaling::T, n::Integer, dim) where {T<:Number}
        new{T}(scaling, n, dim)
    end
end
Base.IndexStyle(::Type{<:ScaledIdentityMPO}) = IndexLinear()
Base.getindex(g::ScaledIdentityMPO{Bool}, i::Integer) = g.data * IdentityMPOsite(g.dim)
Base.getindex(g::ScaledIdentityMPO, i::Integer) = g.data^(1 / length(g)) * IdentityMPOsite(g.dim)
Base.getindex(g::ScaledIdentityMPO, I) = g.data^(1 / length(g)) * fill(IdentityMPOsite(g.dim), length(I))
data(g::ScaledIdentityMPO) = g.data
sites(g::ScaledIdentityMPO) = g[1:length(g)]


IdentityMPO(n) = ScaledIdentityMPO(true, n, 0)
IdentityMPO(n, d) = ScaledIdentityMPO(true, n, d)
Base.length(mpo::ScaledIdentityMPO) = mpo.length
operatorlength(mpo::ScaledIdentityMPO) = mpo.length
LinearAlgebra.ishermitian(mpo::ScaledIdentityMPO) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPO) = data(mpo)' * data(mpo) ≈ 1
Base.:*(x::K, g::ScaledIdentityMPO) where {K<:Number} = ScaledIdentityMPO(x * data(g), length(g), g.dim)
Base.:*(g::ScaledIdentityMPO, x::K) where {K<:Number} = ScaledIdentityMPO(x * data(g), length(g), g.dim)
Base.:/(g::ScaledIdentityMPO, x::K) where {K<:Number} = inv(x) * g
Base.show(io::IO, g::ScaledIdentityMPO) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPO of length ", length(g)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPO) = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityMPO of length ", length(g)))

Base.:(==)(mpo1::ScaledIdentityMPO, mpo2::ScaledIdentityMPO) = data(mpo1) == data(mpo2) && length(mpo1) == length(mpo2) && mpo1.dim == mpo2.dim

MPO(mpo::MPOsite) = MPO([mpo])
MPO(op::Array{<:Number,2}) = MPO(MPOsite(op))
MPO(ops::Vector{Matrix{T}}) where {T} = MPO(map(MPOsite, ops))
MPO(ops::Vector{Array{T,4}}) where {T} = MPO(map(MPOsite, ops))
data(mpo::MPO) = mpo.data

Base.size(mposite::MPOsite) = size(data(mposite))
# Base.length(mpo::MPOsite) = 1
# Base.length(mpo::MPO) = length(data(mpo))
# Base.IndexStyle(::Type{<:MPOsite}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(mpo::MPOsite, i::Integer) = mpo.data[i]
Base.@propagate_inbounds Base.getindex(mpo::AbstractMPO, i::Integer) = mpo.data[i]
#Base.setindex!(mpo::MPOsite, v, I::Vararg{Integer,4}) = (mpo.data[I] = v)
#Base.setindex!(mpo::AbstractMPO, v, I::Vararg{Integer,N}) where {N} = (mpo.data[I] = v)


"""
	auxillerate(mpo)

Return tensor⨂Id_aux
"""
function auxillerate(mpo::MPOsite)
    sop = size(mpo)
    d = sop[2]
    idop = Matrix{eltype(mpo)}(I, d, d)
    @tensor tens[:] := idop[-3, -5] * mpo.data[-1, -2, -4, -6]
    return MPOsite(reshape(tens, sop[1], d^2, d^2, sop[4]))
end

auxillerate(mpo::MPO) = MPO(auxillerate.(mpo.data))
#auxillerate(mpo::HermitianMPO) = HermitianMPO(auxillerate.(mpo.data))

#TODO add density matrix compression of LazyProduct: https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/
struct LazyProduct{MPS<:AbstractMPS,SITE<:AbstractSite,MPO<:AbstractMPO} <: AbstractMPS{SITE}
    mpo::MPO
    mps::MPS
end
Base.:*(mpo::MPO, mps::AbstractMPS{S}) where {MPO<:AbstractMPO,S<:AbstractSite} = LazyProduct{typeof(mps),typeof(mpo[1] * mps[1]),MPO}(mpo, mps)
truncation(lp::LazyProduct) = truncation(lp.mps)
Base.length(lp::LazyProduct) = length(lp.mps)
Base.eltype(lp::LazyProduct) = eltype(lp.mps)
Base.getindex(lp::LazyProduct, i::Integer) = lp.mpo[i] * lp.mps[i]
Base.size(lp::LazyProduct) = size(lp.mps)
#boundaryconditions(::Type{LazyProduct{MPS,S,MPO}}) where {MPS,S,MPO} = boundaryconditions(MPS)
boundaryconditions(mps::LazyProduct) = boundaryconditions(mps.mps)
Base.error(lp::LazyProduct) = error(lp.mps)

function LCROpenMPS(lp::LazyProduct; center = 1, method = :qr)
    Γ = to_left_right_orthogonal(lp[1:end], center = center, method = method)
    LCROpenMPS(Γ, truncation = truncation(lp), error = error(lp))
end

struct LazySiteProduct{T,N,S,O} <: AbstractSite{T,N}
    site::S
    ops::O
    function LazySiteProduct(site::AbstractSite{T,N}, ops::Vararg{O,<:Any}) where {O<:AbstractMPOsite,T,N}
        new{T,N,typeof(site),typeof(ops)}(site, ops)
    end
end
Base.show(io::IO, lp::LazySiteProduct) =
    print(io,"LazySiteProduct\nSite: ", typeof(lp.site), "\nOps: ", typeof.(lp.ops))
Base.show(io::IO, m::MIME"text/plain", lp::LazySiteProduct) = show(io, lp)

Base.eltype(::LazySiteProduct{T,<:Any,<:Any,<:Any}) where {T} = T

Base.:*(o::AbstractMPOsite, s::AbstractSite) = LazySiteProduct(s, o)
Base.:*(o::AbstractMPOsite, lp::LazySiteProduct) = LazySiteProduct(lp.site, o, lp.ops...)

function dense(lp::LazySiteProduct)
    foldr(multiply, lp.ops, init = lp.site)
end
function multiply(op::MPOsite, site::GenericSite)
    sop = size(op)
    ss = size(site)
    @tensor out[:] := data(op)[-1, -3, 1, -4] * data(site)[-2, 1, -5]
    GenericSite(reshape(out, sop[1] * ss[1], sop[2], sop[4] * ss[3]), site.purification)
end

# %% TODO: make dense and Lazy mpo sums
struct MPOSiteSum{S<:Tuple,T} <: AbstractMPOsite{T}
    sites::S
    function MPOSiteSum(sites::Tuple)
        new{typeof(sites),promote_type(eltype.(sites)...)}(sites)
    end
end

multiply(op::MPOSiteSum, site::GenericSite) = SiteSum(Tuple([multiply(o, site) for o in sites(op)]))
sites(opsite::MPOsite) = [opsite]
sites(opsite::MPOSiteSum) = opsite.sites

struct MPOSum{MPOs<:Tuple,Num} <: AbstractMPO{Num}
    mpos::MPOs
    scalings::Vector{Num}
    function MPOSum(mpos::Tuple, scalings::Vector{Num}) where {Num}
        @assert all(length.(mpos) .== length(mpos[1])) "Length of mpos is not the same"
        @assert length(mpos) == length(scalings) "Number of mpos don't match the numver of scale factors."
        new{typeof(mpos),Num}(mpos, scalings)
    end
end
function MPOSum(mpos::Tuple)
    Num = promote_type(eltype.(eltype.(mpos))...)
    MPOSum(mpos, fill(one(Num), length(mpos)))
end
Base.show(io::IO, mpo::MPOSum) =
    (print(io, "MPS: ", typeof(mpo), "\nSites: ", eltype(mpo), "\nLength: ", length(mpo), "\nSum of ", length(mpo.mpos), " mpo's\nWith scalings "); show(io, mpo.scalings))
Base.show(io::IO, m::MIME"text/plain", mpo::MPOSum) = show(io, mpo)
Base.size(mpo::MPOSum) = (length(mpo),)
Base.length(mpo::MPOSum) = length(mpo.mpos[1])
Base.copy(mpo::MPOSum) = MPOSum(copy(mpo.mpos), copy(mpo.scalings))

Base.show(io::IO, mposite::MPOSiteSum) =
    print(io, "SiteSum: ", typeof(mposite), "\nSites: ", eltype(mposite), "\nLength: ", length(mposite.sites))
Base.show(io::IO, m::MIME"text/plain", mposite::MPOSiteSum) = show(io, mposite)

Base.size(sites::MPOSiteSum) = (sum(size.(sites.sites, 1)), size(sites.sites[1], 2), size(sites.sites[1], 3), sum(size.(sites.sites, 4)))
Base.length(sites::MPOSiteSum) = length(sites.sites)
Base.size(sites::MPOSiteSum, i::Integer) = size(sites)[i]

Base.conj(site::MPOSiteSum) = MPOSiteSum(conj.(site.sites))

Base.:+(mps1::AbstractMPO, mps2::AbstractMPO) = MPOSum((mps1, mps2))
Base.:+(mpo::AbstractMPO, sum::MPOSum) = 1 * mpo + sum
Base.:+(sum::MPOSum, mpo::AbstractMPS) = sum + 1 * mpo
Base.:+(s1::MPOSum, s2::MPOSum) = MPOSum(tuple(s1.mpos..., s2.mpos...), vcat(s1.scalings, s2.scalings))

#Base.:+(s1::AbstractMPOsite, s2::AbstractMPOsite) = MPOSiteSum((s1, s2))
Base.:+(s1::MPOSiteSum, s2::MPOSiteSum) = MPOSiteSum(tuple(s1.sites..., s2.sites...))
Base.:+(s1::MPOSiteSum, s2::MPOsite) = MPOSiteSum(tuple(s1.sites..., s2))
Base.:+(s1::MPOsite, s2::MPOSiteSum) = s2 + s1
Base.:+(s1::MPOsite, s2::MPOsite) = MPOSiteSum((s1, s2))

Base.:*(x::Number, mpo::AbstractMPO) = MPSSum((mpo,), [x])
Base.:*(mpo::AbstractMPO, x::Number) = MPSSum((mpo,), [x])
Base.:*(x::Number, mpo::MPOSum) = MPOSum(mpo.mpos, x * mps.scalings)
Base.:*(mpo::MPOSum, x::Number) = x * mpo
Base.:/(mpo::MPOSum, x::Number) = inv(x) * mpo
Base.:-(mpo::AbstractMPO) = (-1) * mpo
Base.:-(mpo::AbstractMPO, mpo2::AbstractMPO) = mpo + (-1) * mpo2

Base.IndexStyle(::Type{<:MPOSum}) = IndexLinear()
Base.getindex(sum::MPOSum, i::Integer) = MPOSiteSum(map(mpo -> mpo[i], sum.mpos))

Base.getindex(sum::MPOSiteSum, i::Integer) = sum.sites[i]
Base.IndexStyle(::Type{<:MPOSiteSum}) = IndexLinear()
reverse_direction(sitesum::MPOSiteSum) = MPOSiteSum(reverse_direction.(sitesum.sites))

Base.eltype(sum::MPOSum) = typeof(sum[1])

dense(site::MPOsite) = site
function dense(site::ScaledIdentityMPOsite{T}) where {T}
    tensor = zeros(T, 1, site.dim, site.dim, 1)
    tensor[1, :, :, 1] = Matrix(data(site) * I, site.dim, site.dim)
    MPOsite(tensor)
end
function dense(sitesum::MPOSiteSum{Tup,T}) where {Tup,T}
    sites = dense.(sitesum.sites)
    sizes = size.(sitesum.sites)
    d = sizes[1][2] #Maybe check that all sites have the same physical dim?
    DL = sum([s[1] for s in sizes])
    DR = sum([s[4] for s in sizes])
    newsite = zeros(T, DL, d, d, DR)
    lastL = 0
    lastR = 0
    for (site, size) in zip(sites, sizes)
        nextL = lastL + size[1]
        nextR = lastR + size[4]
        newsite[lastL+1:nextL, :, :, lastR+1:nextR] = data(site)
        lastL = nextL
        lastR = nextR
    end
    return MPOsite(newsite)
end

dense(mpo::AbstractMPO) = MPO(dense.(mpo), boundary(mpo))
function dense(mpo::MPOSum)
    sites = dense.(mpo)
    scalings = reshape(mpo.scalings, 1, length(mpo.scalings))
    @tensor left[:] := scalings[-1, 1] * data(sites[1])[1, -2, -3, -4]
    sites[1] = MPOsite(left)
    scalings = reshape(mpo.scalings, 1, length(mpo.scalings))
    sites[end] = MPOsite(sum(data(sites[end]), dims = 4))
    MPO(sites)
end


function boundary(::OpenBoundary, mpo::MPO, side::Symbol)
    if side == :right
        return fill(one(eltype(mpo[end])), length(sites(mpo[end]))) # [one(eltype(mpo[end]))]
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return mpo.boundary
    end
end

function boundary(::OpenBoundary, mpo::MPOSum, side::Symbol)
    if side == :right
        return BlockBoundaryVector([boundary(OpenBoundary(), m, :right) for m in mpo.mpos])
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return BlockBoundaryVector([s * boundary(OpenBoundary(), m, :left) for (s, m) in zip(mpo.scalings, mpo.mpos)])
    end
end
# boundary(mpos::MPOSum) = reduce(vcat, [s*boundary(mpo) for (mpo,s) in zip(mpos.mpos,mpos.scalings)])
# boundary(mpo::MPO) = mpo.boundary
boundary(::ScaledIdentityMPO{T}) where {T} = [one(T)]

"""
gives the mpo corresponding to a*mpo1 + b*mpo2.
"""
function addmpos(mpo1, mpo2, a, b, Dmax, tol = 0) #FIXME
    L = length(mpo1)
    d = size(mpo1[1])[2]
    T = promote_type(eltype(eltype(mpo1)), eltype(eltype(mpo2)))
    mpo = Vector{MPOsite{T}}(undef, L)
    mpo[1] = permutedims(cat(1, permutedims(a * mpo1[1], [4, 1, 2, 3]), permutedims(b * mpo2[1], [4, 1, 2, 3])), [2, 3, 4, 1])
    for i = 2:L-1
        mpo[i] = permutedims([permutedims(mpo1[i], [1, 4, 2, 3]) zeros(size(mpo1[i])[1], size(mpo2[i])[4], d, d); zeros(size(mpo2[i])[1], size(mpo1[i])[4], d, d) permutedims(mpo2[i], [1, 4, 2, 3])], [1, 3, 4, 2])
        if tol > 0 || size(mpo1[i])[3] + size(mpo2[i])[3] > Dmax
            @tensor tmp[-1, -2, -3, -4, -5, -6] := mpo[i-1][-1, -2, -3, 1] * mpo[i][1, -4, -5, -6]
            tmp = reshape(tmp, size(mpo[i-1])[1] * d * d, d * d * size(mpo[i])[4])
            F = svd(tmp)
            U, S, V = truncate_svd(F, TruncationArgs(Dmax, tol, false))
            mpo[i-1] = reshape(1 / 2 * U * Diagonal(S), size(mpo[i-1])[1], d, d, D)
            mpo[i] = reshape(2 * V, D, d, d, size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1, permutedims(mpo1[L], [1, 4, 2, 3]), permutedims(mpo2[L], [1, 4, 2, 3])), [1, 3, 4, 2])
    if tol > 0
        @tensor tmp[-1, -2, -3, -4, -5, -6] := mpo[L-1][-1, -2, -3, 1] * mpo[L][1, -4, -5, -6]
        tmp = reshape(tmp, size(mpo[L-1])[1] * d * d, d * d * size(mpo[L])[4])
        F = svd(tmp)
        U, S, V = truncate_svd(F, TruncationArgs(Dmax, tol, false))
        mpo[L-1] = reshape(1 / 2 * U * Diagonal(S), size(mpo[L-1])[1], d, d, D)
        mpo[L] = reshape(2 * V, D, d, d, size(mpo[L])[4])
    end
    return mpo
end


function multiply(site1::MPOsite, site2::MPOsite)
    s1 = size(site1)
    s2 = size(site2)
    @tensor temp[:] := data(site1)[-1, -3, 1, -5] * data(site2)[-2, 1, -4, -6]
    return MPOsite(reshape(temp, s1[1] * s2[1], s1[2], s2[3], s1[4] * s2[4]))
end
function multiply(site1::MPOSiteSum, site2::MPOSiteSum)
    newsites = Array{MPOsite,2}(undef, length(sites(site1)), length(sites(site2)))
    for (n1, s1) in enumerate(sites(site1))
        for (n2, s2) in enumerate(sites(site2))
            size1 = size(s1)
            size2 = size(s2)
            @tensor temp[:] := data(s1)[-1, -3, 1, -5] * data(s2)[-2, 1, -4, -6]
            newsites[n1, n2] = MPOsite(reshape(temp, size1[1] * size2[1], size1[2], size2[3], size1[4] * size2[4]))
        end
    end
    sum(newsites)
end

Base.copy(sitesum::MPOSiteSum) = MPOSiteSum(copy.(sitesum.sites))

"""
    multiplyMPOs(mpo1,mpo2)

Naive multiplication of mpos. Multiplies the bond dimensions.
"""
multiplyMPOs(mpo1::MPO, mpo2::MPO) = MPO([s1 * s2 for (s1, s2) in zip(mpo1, mpo2)], kron(boundary(OpenBoundary(), mpo1, :left), boundary(OpenBoundary(), mpo2, :left)))
multiplyMPOs(mpo1::MPOSum, mpo2::MPOSum) = MPOSum(Tuple([multiplyMPOs(m1, m2) for (m1, m2) in Base.product(mpo1.mpos, mpo2.mpos)]), vec([s1 * s2 for (s1, s2) in Base.product(mpo1.scalings, mpo2.scalings)]))
#multiplyMPOs(mpo1::MPOSum, mpo2::MPO) = MPO([s1 * s2 for (s1, s2) in zip(mpo1, mpo2)], kron(mpo1.boundary,mpo2.boundary))
transfer_matrix_bond(::AbstractMPOsite) = I
# transfer_matrix_bond(mpo::MPOsite{T}) where T = IdentityTransferMatrix(T,(size(mpo,1),size(mpo,4)))
# transfer_matrix_bond(mpo::MPOSiteSum{T}) where T = IdentityTransferMatrix(T,(blocksizes(size(mpo,1)),(blocksizes(size(mpo,4)))


function Matrix(mpo::MPO)
    n = length(mpo)
    T = eltype(mpo[1])
    tens = SparseArray(ones(T, 1, 1, 1))
    for site in mpo[1:n]
        dat = SparseArray(data(site))
        @tensor tens[out, newout, in, newin, right] := tens[out, in, c] * dat[c, newout, newin, right]
        st = size(tens)
        tens = SparseArray(reshape(tens, st[1] * st[2], st[3] * st[4], st[5]))
    end
    return tens[:, :, 1]
end