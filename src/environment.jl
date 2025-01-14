abstract type AbstractEnvironment end
abstract type AbstractInfiniteEnvironment <: AbstractEnvironment end
abstract type AbstractFiniteEnvironment <: AbstractEnvironment end

struct DenseFiniteEnvironment{T,N} <: AbstractFiniteEnvironment
    L::Vector{Array{T,N}}
    R::Vector{Array{T,N}}
end
struct DenseInfiniteEnvironment{T,N} <: AbstractInfiniteEnvironment
    L::Vector{Array{T,N}}
    R::Vector{Array{T,N}}
end

# struct ScaledIdentityEnvironment{T} <: AbstractEnvironment
#     scaling::T
#     function ScaledIdentityEnvironment(scaling::T) where {T<:Number}
#         new{T}(scaling)
#     end
# end
# const IdentityEnvironment = IdentityEnvironment(true)


Base.length(env::AbstractEnvironment) = length(env.L)
finite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseFiniteEnvironment(L, R)
infinite_environment(L::Vector{Array{T,N}}, R::Vector{Array{T,N}}) where {T,N} = DenseInfiniteEnvironment(L, R)

function halfenvironment(t1::Tuple, t2::Tuple, dir::Symbol)
    #T = numtype(mps1)
    Ts = transfer_matrices(t1, t2, reverse_direction(dir))
    V = boundary(t1, t2, dir)
    N = length(t1[1])
    env = Vector{Array{eltype(V),length(t1) + length(t2)}}(undef, N)
    if dir == :left
        itr = 1:1:N
        sizes = [(map(s -> size(s, 1), _sites_from_tuple(t1, k))..., map(s -> size(s, 1), _sites_from_tuple(t2, k))...) for k in 1:N]
    elseif dir == :right
        itr = N:-1:1
        sizes = [(map(s -> size(s)[end], _sites_from_tuple(t1, k))..., map(s -> size(s)[end], _sites_from_tuple(t2, k))...) for k in 1:N]
    else
        @error "In environment: choose direction :left or :right"
    end
    #Vsize(k) = (size(mps1[k],s1), size(mpo[k],s2), size(mps2[k],s1))
    for k in itr
        env[k] = reshape(V, sizes[k])
        if k != itr[end]
            V = Ts[k] * V
        end
    end
    return env
end

# function halfenvironment(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, dir::Symbol)
#     T = numtype(mps1)
#     Ts = transfer_matrices(mps1, mpo, mps2, reverse_direction(dir))
#     V = boundary((mps1,), (mpo, mps2), dir)
#     N = length(mps1)
#     env = Vector{Array{T,3}}(undef, N)
#     if dir == :left
#         itr = 1:1:N
#         s1 = 1
#         s2 = 1
#     elseif dir == :right
#         itr = N:-1:1
#         s1 = 3
#         s2 = 4
#     else
#         @error "In environment: choose direction :left or :right"
#     end
#     #Vsize(k) = (size(mps1[k],s1), size(mpo[k],s2), size(mps2[k],s1))
#     for k in itr
#         env[k] = reshape(V, size(mps1[k], s1), size(mpo[k], s2), size(mps2[k], s1))
#         if k != itr[end]
#             V = Ts[k] * V
#         end
#     end
#     return env
# end
# function halfenvironment(mps1::AbstractMPS, mpo::ScaledIdentityMPO, mps2::AbstractMPS, dir::Symbol)
#     T = numtype(mps1)
#     Ts = data(mpo) * transfer_matrices(mps1, mps2, reverse_direction(dir))
#     V::Vector{T} = boundary((mps1,), (mpo, mps2), dir)
#     N = length(mps1)
#     env = Vector{Array{T,2}}(undef, N)
#     if dir == :left
#         itr = 1:1:N
#         s = 1
#     elseif dir == :right
#         itr = N:-1:1
#         s = 3
#     else
#         @error "In environment: choose direction :left or :right"
#     end
#     #Vsize(k) = (size(mps1[k],s), size(mps2[k],s))
#     for k in itr
#         env[k] = reshape(V, size(mps1[k], s), size(mps2[k], s))
#         if k != itr[end]
#             V = Ts[k] * V
#         end
#     end
#     return env
# end

# halfenvironment(mps1::AbstractMPS, mps2::AbstractMPS, dir::Symbol) = halfenvironment(mps1, IdentityMPO(length(mps1)), mps2, dir)
# halfenvironment(mps::AbstractMPS, mpo::AbstractMPO, dir::Symbol) = halfenvironment(mps, mpo, mps, dir)
# halfenvironment(mps::AbstractMPS, dir::Symbol) = halfenvironment(mps, IdentityMPO(length(mps)), mps, dir)

function environment(t1::Tuple, t2::Tuple)
    L = halfenvironment(t1, t2, :left)
    R = halfenvironment(t1, t2, :right)
    if isinfinite(t1[1])
        return infinite_environment(L, R)
    else
        return finite_environment(L, R)
    end
end
# function environment(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS)
#     L = halfenvironment(mps1, mpo, mps2, :left)
#     R = halfenvironment(mps1, mpo, mps2, :right)
#     if isinfinite(mps1)
#         return infinite_environment(L, R)
#     else
#         return finite_environment(L, R)
#     end
# end

environment(mps1::AbstractMPS, mps2::AbstractMPS) = environment((mps1,), (mps2,))
environment(mps::AbstractMPS, mpo::AbstractMPO) = environment((mps,), (mpo, mps))
environment(mps::AbstractMPS) = environment((mps,), (mps,))

# function update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mpo::AbstractMPOsite, mps2::AbstractSite, site::Integer)
#     site == length(env) || (env.L[site+1] = reshape(transfer_matrix(mps1, mpo, mps2, :right)*vec(env.L[site]), size(mps1,3), size(mpo,4), size(mps2,3)))
#     site==1 || (env.R[site-1] = reshape(transfer_matrix(mps1, mpo, mps2, :left)*vec(env.R[site]), size(mps1,1), size(mpo,1), size(mps2,1)))
#     return 
# end

# function update_left_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
#     # sl = [size(site)[end] for site in sites]
#     sl = map(s -> size(s)[end], sites)
#     j == length(env) || (env.L[j+1] = reshape(_local_transfer_matrix(sites, :right) * vec(env.L[j]), sl...))
#     return
# end
# function update_right_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
#     #sr = [size(site)[1] for site in sites]
#     sr = size.(sites, 1)
#     if j > 1
#         env.R[j-1] = reshape(_local_transfer_matrix(sites, :left) * vec(env.R[j]), sr...)
#     end
#     return
# end
# function update_environment!(env::AbstractFiniteEnvironment, j::Integer, sites::Vararg{Union{AbstractSite,AbstractMPOsite},N}) where {N}
#     sl = map(s -> size(s)[end], sites)
#     sr = size.(sites, 1)
#     j == length(env) || (env.L[j+1] = reshape(_local_transfer_matrix(sites, :right) * vec(env.L[j]), sl...))
#     j == 1 || (env.R[j-1] = reshape(_local_transfer_matrix(sites, :left) * vec(env.R[j]), sr...))
#     return
# end
function update_left_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::Tuple, sites::Tuple)
    s = (map(s -> size(s)[end], csites)..., map(s -> size(s)[end], sites)...)
    if j < length(env)
        env.L[j+1] = reshape(_local_transfer_matrix(csites, sites, :right) * vec(env.L[j]), s)
    end
    return
end
function update_right_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::Tuple, sites::Tuple)
    s = (map(s -> size(s, 1), csites)..., map(s -> size(s, 1), sites)...)
    if j > 1
        env.R[j-1] = reshape(_local_transfer_matrix(csites, sites, :left) * vec(env.R[j]), s)
    end
    return
end
function update_environment!(env::AbstractFiniteEnvironment, j::Integer, csites::Tuple, sites::Tuple)
    update_left_environment!(env, j, csites, sites)
    update_right_environment!(env, j, csites, sites)
    # j == length(env) || (env.L[j+1] = _local_transfer_matrix(csites, sites, :right) * env.L[j])
    # j == 1 || (env.R[j-1] = _local_transfer_matrix(csites, sites, :left) * env.R[j])
    # return
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
local_mul!(out, envL, envR, mposite::AbstractMPOsite, site::AbstractArray{<:Number,3}) = @tensor out[:] = (envL[-1, 2, 3] * data(mposite)[2, -2, 4, 5]) * (site[3, 4, 1] * envR[-3, 5, 1])

local_mul(envL, envR, mposite::AbstractMPOsite, site::AbstractArray{<:Number,3}) = @tensor temp[:] := (envL[-1, 2, 3] * data(mposite)[2, -2, 4, 5]) * (site[3, 4, 1] * envR[-3, 5, 1])
local_mul(envL, envR, mposite::AbstractMPOsite, site::GenericSite) = GenericSite(local_mul(envL, envR, mposite, data(site)), ispurification(site))
local_mul(envL, envR, mposite::AbstractMPOsite, site::OrthogonalLinkSite) = local_mul(envL, envR, mposite, site.Λ1 * site * site.Λ2)
#TODO implement these for SiteSum
local_mul(envL, envR, site::Array{<:Number,3}) = @tensor temp[:] := envL[-1, 1] * site[1, -2, 2] * envR[-3, 2]
local_mul(envL, envR, site::GenericSite) = GenericSite(local_mul(envL, envR, data(site)), ispurification(site))
local_mul(envL, envR, site::OrthogonalLinkSite) = local_mul(envL, envR, site.Λ1 * site.Γ * site.Λ2)

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