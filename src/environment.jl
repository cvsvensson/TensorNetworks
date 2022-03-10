abstract type AbstractEnvironment end
abstract type AbstractInfiniteEnvironment <: AbstractEnvironment end
abstract type AbstractFiniteEnvironment <: AbstractEnvironment end

abstract type AbstractBoundaryVector{T,N} <: AbstractVector{T} end
struct BoundaryVector{T,N} <: AbstractBoundaryVector{T,N}
    data::Array{T,N}
end
BoundaryVector(x::Number) = BoundaryVector([x]) 
struct BlockBoundaryVector{T,N} <: AbstractBoundaryVector{T,N}
    data::Array{BoundaryVector{T,N},N}
end
Base.vec(bv::Union{BlockBoundaryVector,BoundaryVector}) = vec(bv.data)
Base.length(bv::Union{BoundaryVector}) = length(data(bv))
Base.length(bv::Union{BlockBoundaryVector}) = prod(length.(data(bv)))
data(bv::Union{BlockBoundaryVector,BoundaryVector}) = bv.data

# Base.iterate(v::Union{BlockBoundaryVector,BoundaryVector}, state) = iterate(data(v),state)
# Base.iterate(v::Union{BlockBoundaryVector,BoundaryVector}) = iterate(data(v))
# Base.IteratorSize(v::Union{BlockBoundaryVector,BoundaryVector}) = Base.IteratorSize(data(v))
Base.size(v::Union{BlockBoundaryVector,BoundaryVector}) = (length(v),)
Base.size(v::Union{BlockBoundaryVector,BoundaryVector},i) = i==1 ? length(v) : 1
Base.eltype(::BlockBoundaryVector{T,N}) where {T,N} = BoundaryVector{T,N}
Base.eltype(::BoundaryVector{T,N}) where {T,N} = T

Base.similar(v::BlockBoundaryVector) = BlockBoundaryVector(similar(data(v)))
Base.similar(v::BoundaryVector) = BoundaryVector(similar(data(v)))
Base.similar(v::BlockBoundaryVector, dims::Dims) = BlockBoundaryVector(similar(data(v),dims))
Base.similar(v::BoundaryVector, dims::Dims) = BoundaryVector(similar(data(v),dims))
# Base.similar(v::BoundaryVector, s::Type{S}) where S = similar(data(v),s)
Base.copy(v::BlockBoundaryVector) = BlockBoundaryVector(copy.(v))
Base.copy(v::BoundaryVector) = BoundaryVector(copy(data(v)))
# Base.copyto!(dest::BlockBoundaryVector, v::BlockBoundaryVector) = BlockBoundaryVector(copy.(data(v)))
# Base.copyto!(dest::BoundaryVector, v::BoundaryVector) = BoundaryVector(copy(data(v)))
Base.getindex(v::Union{BlockBoundaryVector,BoundaryVector}, x::Vararg{Int, N}) where N = getindex(data(v), x...)
Base.setindex!(v::Union{BlockBoundaryVector,BoundaryVector}, value, i::Vararg{Int, N}) where N = setindex!(data(v),value, i...)
# Base.ndims(v::BlockBoundaryVector{<:Number,N}) where {N} = N
# Base.ndims(v::BoundaryVector{<:Number,N}) where {N} = N

LinearAlgebra.dot(v::BoundaryVector, w::BoundaryVector) = dot(data(v), data(w))
LinearAlgebra.norm(v::BoundaryVector) = norm(data(v))

LinearAlgebra.dot(v::BlockBoundaryVector, w::BlockBoundaryVector) = mapreduce(dot, +, v.data, w.data)
LinearAlgebra.norm(v::BlockBoundaryVector) = sqrt(mapreduce(x -> norm(x)^2, +, v))

 
function BlockBoundaryVector(v::BoundaryVector{T,N}) where {T<:Number,N}
    bv = Array{BoundaryVector{T,N},N}(undef, (1 for k in 1:N)...)
    bv[1] = v
    return BlockBoundaryVector(bv)
end
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
function BlockBoundaryVector(vecs::Vararg{BlockBoundaryVector,N}) where {N}
    BlockBoundaryVector([BoundaryVector(bvs...) for bvs in Base.product(data.(vecs)...)])
end
BoundaryVector(bvs::Vararg{BoundaryVector,N}) where N = BoundaryVector(tensor_product(data.(bvs)...))
BlockBoundaryVector(bvs::Vararg{Union{BoundaryVector,BlockBoundaryVector},N}) where N = BlockBoundaryVector(BlockBoundaryVector.(bvs)...)

tensor_product(t1::AbstractArray) = t1
tensor_product(t1::AbstractArray{T1,N1}, t2::AbstractArray{T2,N2}) where {N1,N2,T1,T2} = tensorproduct(t1, 1:N1, t2, N1 .+ (1:N2))::Array{promote_type(T1, T2),N1 + N2}
tensor_product(tensors::Vararg{<:AbstractArray,N}) where N =  reduce(tensor_product, tensors)

struct DenseFiniteEnvironment{V,T,N} <: AbstractFiniteEnvironment
    L::Vector{V}
    R::Vector{V}
    #leftblocksizes::Vector{Array{NTuple{N,Int},K}}
    #rightblocksizes::Vector{Array{NTuple{N,Int},K}}
    function DenseFiniteEnvironment(L::Vector{<:AbstractBoundaryVector{T,N}}, R::Vector{<:AbstractBoundaryVector{T,N}}) where {T,N}
        new{eltype(L),T,N}(L,R)
    end
end
struct DenseInfiniteEnvironment{V,T,N} <: AbstractInfiniteEnvironment
    L::Vector{V}
    R::Vector{V}
    function DenseInfiniteEnvironment(L::Vector{<:AbstractBoundaryVector{T,N}}, R::Vector{<:AbstractBoundaryVector{T,N}}) where {T,N}
        new{eltype(L),T,N}(L,R)
    end
end

Base.length(env::AbstractEnvironment) = length(env.L)
#finite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseFiniteEnvironment(L, R)
#infinite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseInfiniteEnvironment(L, R)
finite_environment(L::Vector{AbstractBoundaryVector{T,N}}, R::Vector{AbstractBoundaryVector{T,N}}) where {T,N} = DenseFiniteEnvironment(L,R)
infinite_environment(L::Vector{AbstractBoundaryVector{T,N}}, R::Vector{AbstractBoundaryVector{T,N}}) where {T,N} = DenseInfiniteEnvironment(L, R)


function halfenvironment(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, dir::Symbol)
    T = numtype(mps1, mps2)
    Ts = transfer_matrices(mps1, mpo, mps2, reverse_direction(dir))
    V = boundary(mps1, mpo, mps2, dir)
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
    T = numtype(mps1)
    Ts = data(mpo) * transfer_matrices(mps1, mps2, reverse_direction(dir))
    V::Vector{T} = vec(boundary(mps1, mps2, dir))
    N = length(mps1)
    env = Vector{Array{T,2}}(undef, N)
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
        env[k] = reshape(V, size(mps1[k], s), size(mps2[k], s))
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

function update_left_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
    # sl = [size(site)[end] for site in sites]
    sl = map(s -> size(s)[end], sites)
    j == length(env) || (env.L[j+1] = reshape(_local_transfer_matrix(sites, :right) * vec(env.L[j]), sl...))
    return
end
function update_right_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
    #sr = [size(site)[1] for site in sites]
    sr = size.(sites, 1)
    if j > 1
        env.R[j-1] = reshape(_local_transfer_matrix(sites, :left) * vec(env.R[j]), sr...)
    end
    return
end
function update_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
    sl = map(s -> size(s)[end], sites)
    sr = size.(sites, 1)
    j == length(env) || (env.L[j+1] = reshape(_local_transfer_matrix(sites, :right) * vec(env.L[j]), sl...))
    j == 1 || (env.R[j-1] = reshape(_local_transfer_matrix(sites, :left) * vec(env.R[j]), sr...))
    return
end

update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mps2::AbstractSite, site::Integer) = update_environment!(env, site, mps1, mps2)
update_environment!(env::AbstractFiniteEnvironment, mps::AbstractSite, mpo::AbstractMPOsite, site::Integer) = update_environment!(env, site, mps, mpo, mps)
update_environment!(env::AbstractFiniteEnvironment, mps::AbstractSite, site::Integer) = update_environment!(env, site, mps, mps)

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

#Only works for certain environs
function local_mul(envL, envR, sitesum::SiteSum)
    println(size(envL))
    println(size(envR))
    println(size.(sites(sitesum)))
    leftsizes = size.(sites(sitesum), 1)
    leftsizematrix = collect(Base.product(leftsizes, leftsizes))
    lefttensors = _split_vector(vec(envL), leftsizematrix)
    rightsizematrix = collect(Base.product(size.(sites(sitesum), 3), size.(sites(sitesum), 3)))
    righttensors = _split_vector(vec(envR), rightsizematrix)
    @assert length(lefttensors) == length(righttensors) == length(sites(sitesum))^2
    arrays = sum([data(local_mul(lefttensors[n1, n2], righttensors[n1, n2], sites(sitesum)[n2])) for (n1, n2) in Base.product(1:length(sites(sitesum)), 1:length(sites(sitesum)))], dims = 2)
    SiteSum(Tuple(GenericSite.(arrays, ispurification(sitesum))))

    #SiteSum(map((L,R,s) -> local_mul(L, R, s), lefttensors, righttensors, sites(sitesum)))

    #SiteSum(Tuple(local_mul(L, R, s) for (L,s,R) in zip(lefttensors,sites(sitesum),righttensors)))
    #FIXME: Test if this works
end

function _apply_transfer_matrices(Ts)
    # N1, N2 = size(Ts)
    # # sizes = [size(Γ1[n1],3)*size(Γ1[n2],3) for n1 in 1:N1, n2 in 1:N1]
    #sizes = [size(T,1) for T in Ts]
    DL = sum(size.(Ts, 1))
    DR = sum(size.(Ts, 2))
    function f(v::BlockBoundaryVector)
        # tens = _split_vector(vec(v), sizes)
        #_join_tensor([T * t for (T, t) in zip(Ts, tens)])
        BlockBoundaryVector([ T * t for (T, t) in zip(Ts, data(v))])
    end
    function f_adjoint(v::BlockBoundaryVector)
        #tens = _split_vector(vec(v), sizes)
        #_join_tensor([T' * t for (T, t) in zip(Ts, tens)])
        BlockBoundaryVector([T' * t for (T, t) in zip(Ts, data(v))])
    end
    return LinearMap{eltype(Ts[1])}(f, f_adjoint, DL, DR)
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
# Base.getindex(env::ScaledIdentityEnvironment,i::Integer, dir::Symbol) = env