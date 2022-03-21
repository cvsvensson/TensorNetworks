
# abstract type AbstractBoundaryVector{T,N} <: AbstractArray{T,N} end
# struct BoundaryVector{T,N} <: AbstractBoundaryVector{T,N}
#     data::Array{T,N}
# end
# BoundaryVector(x::Number) = BoundaryVector([x])

Base.vec(bv::BlockBoundaryVector) = reduce(vcat, vec.(bv.data))
Base.conj(bv::BlockBoundaryVector) = BlockBoundaryVector(conj(data(bv)))
#Base.convert(::Type{BlockBoundaryVector{T,N}},m::Array{<:Any,N}) where {T,N} = BlockBoundaryVector(m)

# Base.length(bv::Union{BoundaryVector}) = length(data(bv))
# Base.length(bv::Union{BlockBoundaryVector}) = prod(length.(data(bv)))
data(bv::Union{BlockBoundaryVector}) = bv.data
# BlockBoundaryVector(v::Array{T,N}) where {T,N} = BlockBoundaryVector(v)

Base.:*(x::Number, v::T) where {T<:Union{BlockBoundaryVector}} = BlockBoundaryVector(x * data(v))
Base.:*(v::T, x::Number) where {T<:Union{BlockBoundaryVector}} = x * v
Base.:/(v::T, x::Number) where {T<:Union{BlockBoundaryVector}} = inv(x) * v

# Base.iterate(v::Union{BlockBoundaryVector,BoundaryVector}, state) = iterate(data(v),state)
# Base.iterate(v::Union{BlockBoundaryVector,BoundaryVector}) = iterate(data(v))
# Base.IteratorSize(v::Union{BlockBoundaryVector,BoundaryVector}) = Base.IteratorSize(data(v))
Base.size(v::Union{BlockBoundaryVector}) = size(data(v))
Base.size(v::Union{BlockBoundaryVector}, i) = size(data(v), i) #i == 1 ? length(v) : 1
# Base.eltype(::BlockBoundaryVector{T,N}) where {T,N} = Array{T,N}

# Base.reshape(v::BoundaryVector, dims::Dims) = BoundaryVector(reshape(v.data,dims))

# Base.similar(v::B) where {B<:Union{BlockBoundaryVector,BoundaryVector}} = B(similar(data(v)))
# Base.similar(v::BoundaryVector) = BoundaryVector(similar(data(v)))
# Base.similar(v::B, dims::Dims) where {B<:Union{<:BlockBoundaryVector,<:BoundaryVector}} = B(similar(data(v), dims))
# Base.similar(v::BoundaryVector, dims::Dims) = BoundaryVector(similar(data(v), dims))
# Base.similar(v::B, ::Type{S}) where {S,B<:Union{BlockBoundaryVector,BoundaryVector}} = B(similar(data(v), S))
# Base.similar(v::BoundaryVector, ::Type{S}) where S = BoundaryVector(similar(data(v), S))
# Base.similar(v::BoundaryVector, ::Type{S}, dims::Dims) where S = BoundaryVector(similar(data(v), S, dims))
Base.similar(v::BlockBoundaryVector) = BlockBoundaryVector(similar.(data(v)))
# Base.similar(v::BlockBoundaryVector, dims::Dims) = BlockBoundaryVector(similar(data(v), dims))
Base.similar(v::BlockBoundaryVector, ::Type{S}) where {S} = BlockBoundaryVector(similar(data(v), S))
Base.similar(v::BlockBoundaryVector, ::Type{S}, dims::Dims) where {S} = BlockBoundaryVector(similar(data(v), S, dims))

# Base.similar(v::B, ::Type{S}) where {S,B<:Union{BlockBoundaryVector,BoundaryVector}} = B(similar(data(v), S))
# Base.similar(v::BoundaryVector, ::Type{S}) where S = BoundaryVector(similar(data(v), S))
# Base.similar(v::B, ::Type{S}, dims::Dims) where {S,B<:Union{<:BlockBoundaryVector,<:BoundaryVector}} = B(similar(data(v), S, dims))
# Base.similar(v::BoundaryVector, s::Type{S}) where S = similar(data(v),s)
Base.copy(v::B) where {B<:Union{BlockBoundaryVector}} = B(copy.(v))
# Base.copy(v::BoundaryVector) = BoundaryVector(copy(data(v)))
# Base.copyto!(dest::BlockBoundaryVector, v::BlockBoundaryVector) = BlockBoundaryVector(copy.(data(v)))
# Base.copyto!(dest::BoundaryVector, v::BoundaryVector) = BoundaryVector(copy(data(v)))
Base.getindex(v::Union{BlockBoundaryVector}, x::Vararg{Int,N}) where {N} = getindex(data(v), x...)
Base.setindex!(v::Union{BlockBoundaryVector}, value, i::Vararg{Int,N}) where {N} = setindex!(data(v), value, i...)
# Base.ndims(v::BlockBoundaryVector{<:Number,N}) where {N} = N
# Base.ndims(v::BoundaryVector{<:Number,N}) where {N} = N

# LinearAlgebra.dot(v::BoundaryVector, w::BoundaryVector) = dot(data(v), data(w))
# LinearAlgebra.norm(v::BoundaryVector) = norm(data(v))

LinearAlgebra.dot(v::BlockBoundaryVector, w::BlockBoundaryVector) = mapreduce(dot, +, v.data, w.data)
LinearAlgebra.norm(v::BlockBoundaryVector) = sqrt(mapreduce(x -> norm(x)^2, +, v))

#TODO: Implement the rest of the operations necessary for eigsolve
###
# LinearAlgebra.mul!(w::BoundaryVector, v::BoundaryVector, x::Number) = BoundaryVector(mul!(data(w), data(v),x))
# LinearAlgebra.mul!(C::BoundaryVector, A, B, α, β) )
# LinearAlgebra.rmul!(v::BoundaryVector, x::Number) = BoundaryVector(rmul!(data(v), x))
LinearAlgebra.mul!(w::BlockBoundaryVector, v::BlockBoundaryVector, x::Number) = BlockBoundaryVector([mul!(ww, vv, x) for (ww, vv) in zip(data(w), data(v))])
LinearAlgebra.rmul!(v::BlockBoundaryVector, x::Number) = BlockBoundaryVector(rmul!(data(v), x))
function LinearAlgebra.axpy!(x::Number, v::BlockBoundaryVector, w::BlockBoundaryVector)
    BlockBoundaryVector([axpy!(x, sv, sw) for (sv, sw) in zip(data(v), data(w))])
end
function LinearAlgebra.axpby!(x::Number, v::BlockBoundaryVector, β::Number, w::BlockBoundaryVector)
    BlockBoundaryVector([axpby!(x, sv, β, sw) for (sv, sw) in zip(data(v), data(w))])
end

###

# function BoundaryVector(array::Array{T,N})
#     a = Array{T,N}(undef,1,1,1)
# end
# function BlockBoundaryVector(vecs::Vararg{BlockBoundaryVector{<:Number,1},N}) where {N}
#     tens = ones((1 for k in 1:N)...)
#     BlockBoundaryVector(map(x -> prod(x) * tens, Base.product(vecs...)))
# end
#Base.getindex(v::Union{BlockBoundaryVector,BoundaryVector}, x) = getindex(vec(v), x)

#Base.size(v::Union{BlockBoundaryVector,BoundaryVector}) = size(vec(v))
# BlockBoundaryVector(vec::Vector{<:Number}) = BlockBoundaryVector(BoundaryVector)
BlockBoundaryVector(bv::BlockBoundaryVector) = bv
# function tensor_product(vecs::Vararg{BlockBoundaryVector,N}) where {N}
#     BlockBoundaryVector([tensor_product(bvs...) for bvs in Base.product(data.(vecs)...)])
# end
function tensor_product(vecs::Vararg{Union{BlockBoundaryVector,AbstractArray{<:Number,<:Any}},N}) where {N}
    BlockBoundaryVector([tensor_product(bvs...) for bvs in Base.product(data.(BlockBoundaryVector.(vecs))...)])
end
# BoundaryVector(bvs::Vararg{Array,N}) where {N} = BoundaryVector(tensor_product(data.(bvs)...))
# BlockBoundaryVector(bvs::Vararg{Array{T,N},N}) where {T<:Number,N} = BlockBoundaryVector(BlockBoundaryVector.(bvs)...)

BlockBoundaryVector(v, s::Array{NTuple{N,Int},K}) where {N,K} = BlockBoundaryVector(_split_vector(vec(v), s))

tensor_product(t1::AbstractArray) = t1
tensor_product(t1::AbstractArray{T1,N1}, t2::AbstractArray{T2,N2}) where {N1,N2,T1<:Number,T2<:Number} = tensorproduct(t1, 1:N1, t2, N1 .+ (1:N2))::Array{promote_type(T1, T2),N1 + N2}
tensor_product(tensors::Vararg{AbstractArray{<:Number,<:Any},N}) where {N} = foldl(tensor_product, tensors)
tensor_product(t1::UniformScaling, t2::AbstractArray{T2,N2}) where {N2,T2} = t1.λ * t2
tensor_product(t1::AbstractArray{T1,N1}, t2::UniformScaling) where {N1,T1} = t2.λ * t1

struct FiniteEnvironment{V} <: AbstractFiniteEnvironment
    L::Vector{V}
    R::Vector{V}
    #leftblocksizes::Vector{Array{NTuple{N,Int},K}}
    #rightblocksizes::Vector{Array{NTuple{N,Int},K}}
    function FiniteEnvironment(L::Vector{V}, R::Vector{V}) where {V}
        new{V}(L, R)
    end
end
struct InfiniteEnvironment{V} <: AbstractInfiniteEnvironment
    L::Vector{V}
    R::Vector{V}
    function InfiniteEnvironment(L::Vector{V}, R::Vector{V}) where {V}
        new{V}(L, R)
    end
end

Base.length(env::AbstractEnvironment) = length(env.L)
#finite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseFiniteEnvironment(L, R)
#infinite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseInfiniteEnvironment(L, R)
finite_environment(L::Vector{V}, R::Vector{V}) where {V} = FiniteEnvironment(L, R)
infinite_environment(L::Vector{V}, R::Vector{V}) where {V} = InfiniteEnvironment(L, R)


function halfenvironment(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, dir::Symbol)
    T = numtype(mps1, mps2)
    Ts = transfer_matrices((mps1,), (mpo, mps2), reverse_direction(dir))
    V = boundary((mps1,), (mpo, mps2), dir)
    N = length(mps1)
    env = Vector{typeof(V)}(undef, N)
    if dir == :left
        itr = 1:1:N
        s1 = 1
        s2 = 1
    elseif dir == :right
        itr = N:-1:1
        s1 = 3
        s2 = 4
    else
        @error "In environment: choose direction :left or :right"
    end
    #Vsize(k) = (size(mps1[k],s1), size(mpo[k],s2), size(mps2[k],s1))
    for k in itr
        #env[k] = reshape(V, size(mps1[k], s1), size(mpo[k], s2), size(mps2[k], s1))
        # println(typeof(V))
        # println(typeof(env))
        env[k] = V
        if k != itr[end]
            V = Ts[k] * V
        end
    end
    return env
end
function halfenvironment(mps1::AbstractMPS, mpo::ScaledIdentityMPO, mps2::AbstractMPS, dir::Symbol)
    T = numtype(mps1, mps2)
    Ts = data(mpo) * transfer_matrices((mps1,), (mps2,), reverse_direction(dir))
    V = boundary((mps1,), (mps2,), dir)
    N = length(mps1)
    env = Vector{typeof(V)}(undef, N)
    if dir == :left
        itr = 1:1:N
        s = 1
    elseif dir == :right
        itr = N:-1:1
        s = 3
    else
        @error "In environment: choose direction :left or :right"
    end
    #Vsize(k) = (size(mps1[k],s), size(mps2[k],s))
    for k in itr
        env[k] = V#reshape(V, size(mps1[k], s), size(mps2[k], s))
        if k != itr[end]
            V = Ts[k] * V
        end
    end
    return env
end

halfenvironment(mps1::AbstractMPS, mps2::AbstractMPS, dir::Symbol) = halfenvironment(mps1, IdentityMPO(length(mps1)), mps2, dir)
halfenvironment(mps::AbstractMPS, mpo::AbstractMPO, dir::Symbol) = halfenvironment(mps, mpo, mps, dir)
halfenvironment(mps::AbstractMPS, dir::Symbol) = halfenvironment(mps, IdentityMPO(length(mps)), mps, dir)


function environment(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS)
    L = halfenvironment(mps1, mpo, mps2, :left)
    R = halfenvironment(mps1, mpo, mps2, :right)
    if isinfinite(mps1)
        return infinite_environment(L, R)
    else
        return finite_environment(L, R)
    end
end

environment(mps1::AbstractMPS, mps2::AbstractMPS) = environment(mps1, IdentityMPO(length(mps1)), mps2)
environment(mps::AbstractMPS, mpo::AbstractMPO) = environment(mps, mpo, mps)
environment(mps::AbstractMPS) = environment(mps, IdentityMPO(length(mps)), mps)

# function update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mpo::AbstractMPOsite, mps2::AbstractSite, site::Integer)
#     site == length(env) || (env.L[site+1] = reshape(transfer_matrix(mps1, mpo, mps2, :right)*vec(env.L[site]), size(mps1,3), size(mpo,4), size(mps2,3)))
#     site==1 || (env.R[site-1] = reshape(transfer_matrix(mps1, mpo, mps2, :left)*vec(env.R[site]), size(mps1,1), size(mpo,1), size(mps2,1)))
#     return 
# end

function update_left_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}}, sites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}})
    # sl = [size(site)[end] for site in sites]
    # sl = map(s -> size(s)[end], sites)
    j == length(env) || (env.L[j+1] = _local_transfer_matrix(csites, sites, :right) * env.L[j])
    return
end
function update_right_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}}, sites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}})
    #sr = [size(site)[1] for site in sites]
    sr = size.(sites, 1)
    if j > 1
        env.R[j-1] = _local_transfer_matrix(csites, sites, :left) * env.R[j]
    end
    return
end
function update_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}}, sites::NTuple{<:Any,Union{AbstractSite,AbstractMPOsite}})
    # sl = map(s -> size(s)[end], sites)
    # sr = size.(sites, 1)
    j == length(env) || (env.L[j+1] = _local_transfer_matrix(csites, sites, :right) * env.L[j])
    j == 1 || (env.R[j-1] = _local_transfer_matrix(csites, sites, :left) * env.R[j])
    return
end

update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mps2::AbstractSite, site::Integer) = update_environment!(env, site, (mps1,), (mps2,))
update_environment!(env::AbstractFiniteEnvironment, mps::AbstractSite, mpo::AbstractMPOsite, site::Integer) = update_environment!(env, site, (mps,), (mpo, mps))
update_environment!(env::AbstractFiniteEnvironment, mps::AbstractSite, site::Integer) = update_environment!(env, site, (mps,), (mps,))
update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mpo::AbstractMPOsite, mps2::AbstractSite, site::Integer) = update_environment!(env, site, (mps1,), (mpo, mps2))

# function update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mpo::ScaledIdentityMPOsite, mps2::AbstractSite, site::Integer)
#     env.L[site+1] = reshape(transfer_matrix(mps1, mpo, mps2, :right)*vec(env.L[site]), size(mps1,3), size(mps2,3))
#     env.R[site-1] = reshape(transfer_matrix(mps1, mpo, mps2, :left)*vec(env.R[site]), size(mps1,1), size(mps2,1))
#     return 
# end

#TODO check performance and compare to matrix multiplication and Tullio
local_mul(envL, envR, mposite::MPOsite, site::AbstractArray{<:Number,3}) = @tensor temp[:] := (envL[-1, 2, 3] * data(mposite)[2, -2, 4, 5]) * (site[3, 4, 1] * envR[-3, 5, 1])
local_mul(envL, envR, mposite::MPOsite, site::GenericSite) = GenericSite(local_mul(envL, envR, mposite, data(site)), ispurification(site))
local_mul(envL, envR, mposite::MPOsite, site::OrthogonalLinkSite) = local_mul(envL, envR, mposite, site.Λ1 * site * site.Λ2)

#TODO implement these for SiteSum
local_mul(envL, envR, site::Array{<:Number,3}) = @tensor temp[:] := envL[-1, 1] * site[1, -2, 2] * envR[-3, 2]
local_mul(envL, envR, site::GenericSite) = GenericSite(local_mul(envL, envR, data(site)), ispurification(site))
local_mul(envL, envR, site::OrthogonalLinkSite) = local_mul(envL, envR, site.Λ1 * site.Γ * site.Λ2)

#local_mul(envL, envR, site::SiteSum) = local_mul(envL, envR, dense(site))

function local_mul(envL, envR, mpo, lp::LazySiteProduct)
    local_mul(envL, envR, mpo, lp.sites...)
end
function local_mul(envL, envR, lp::LazySiteProduct)
    local_mul(envL, envR, lp.sites...)
end

function local_mul(envL, envR, sitesum::SiteSum)
    @assert size(data(envL)) == size(data(envR)) "Error: Left and right environments have different dimensions"
    itr = Base.product(1:size(data(envL), 1), 1:size(data(envL), 2))
    arrays = sum([data(local_mul(data(envL)[n1, n2], data(envR)[n1, n2], sites(sitesum)[n2])) for (n1, n2) in itr], dims = 2)
    SiteSum(Tuple(GenericSite.(arrays, ispurification(sitesum))))
end

function local_mul(envL::BlockBoundaryVector{T,3}, envR::BlockBoundaryVector{T,3}, mpo, site::SiteSum) where {T}
    @assert size(data(envL)) == size(data(envR)) "Error: Left and right environments have different dimensions"
    sizes = size(data(envL))
    itr = Base.product((1:s for s in sizes)...)
    arrays = sum([data(local_mul(data(envL)[ns...], data(envR)[ns...], sites(mpo)[ns[2]], sites(site)[ns[3]])) for ns in itr], dims = 2:3)
    SiteSum(Tuple(GenericSite.(arrays, ispurification(site))))
end
function local_mul(envL::BlockBoundaryVector{T,3}, envR::BlockBoundaryVector{T,3}, mpo, site::GenericSite) where {T}
    outsite = local_mul(envL, envR, mpo, SiteSum(site))
    GenericSite(outsite)
end

function _apply_transfer_matrices(Ts::Array{Maps,N}) where {N,Maps}
    f(v::BlockBoundaryVector) = BlockBoundaryVector([T * t for (T, t) in zip(Ts, data(v))])
    f_adjoint(v::BlockBoundaryVector) = BlockBoundaryVector([T' * t for (T, t) in zip(Ts, data(v))])
    return TransferMatrix(f, f_adjoint, promote_type(eltype.(Ts)...), ([size(T, 1) for T in Ts], [size(T, 2) for T in Ts]))
end

function Base.getindex(env::AbstractEnvironment, i::Integer, dir::Symbol)
    if dir == :right
        return env.R[i]
    else
        if dir !== :left
            @warn "Defaulting to dir==:left"
        end
        return env.L[i]
    end
end
