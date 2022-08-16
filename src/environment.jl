abstract type AbstractEnvironment end
# abstract type AbstractInfiniteEnvironment <: AbstractEnvironment end
# abstract type AbstractFiniteEnvironment <: AbstractEnvironment end

struct Environment{T,N,S<:AbstractArray{T,N}} <: AbstractEnvironment
    L::S
    R::S
end
Environment(L::S,R::S) where S = Environment{eltype(S),ndims(S),S}(L,R)
const DenseEnvironment{T,N} = Environment{T,N,Array{T,N}}


struct Environments{T,N,S<:AbstractArray{T,N}} <: AbstractVector{Environment{T,N,S}}
    L::Vector{S}
    R::Vector{S}
end
Base.length(env::Environments) = length(env.L)
Base.getindex(env::Environments,i::Integer) = Environment(env.L[i],env.R[i])

function Base.getindex(env::Environments, i::Integer, dir::Symbol)
    if dir == :right
        return env.R[i]
    else
        if dir !== :left
            @warn "Defaulting to dir==:left"
        end
        return env.L[i]
    end
end
Environments(L::Vector{Environment{T,N,S}}, R::Vector{Environment{T,N,S}}) where {T,N,S} = Environments{T,N,S}(L, R)

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

function environments(t1::Tuple, t2::Tuple)
    L = halfenvironment(t1, t2, :left)
    R = halfenvironment(t1, t2, :right)
    return Environments(L, R)
end

environments(mps1::AbstractMPS, mps2::AbstractMPS) = environments((mps1,), (mps2,))
environments(mps::AbstractMPS, mpo::AbstractMPO) = environments((mps,), (mpo, mps))
environments(mps::AbstractMPS) = environments((mps,), (mps,))

function update_left_environments!(env::Environments, j::Integer, csites::Tuple, sites::Tuple)
    s = (map(s -> size(s)[end], csites)..., map(s -> size(s)[end], sites)...)
    if j < length(env)
        env.L[j+1] = reshape(_local_transfer_matrix(csites, sites, :right) * vec(env.L[j]), s)
    end
    return
end
function update_right_environments!(env::Environments, j::Integer, csites::Tuple, sites::Tuple)
    s = (map(s -> size(s, 1), csites)..., map(s -> size(s, 1), sites)...)
    if j > 1
        env.R[j-1] = reshape(_local_transfer_matrix(csites, sites, :left) * vec(env.R[j]), s)
    end
    return
end
function update_environments!(env::Environments, j::Integer, csites::Tuple, sites::Tuple)
    update_left_environments!(env, j, csites, sites)
    update_right_environments!(env, j, csites, sites)
    # j == length(env) || (env.L[j+1] = _local_transfer_matrix(csites, sites, :right) * env.L[j])
    # j == 1 || (env.R[j-1] = _local_transfer_matrix(csites, sites, :left) * env.R[j])
    # return
    return
end

update_environments!(env::Environments, mps1::AbstractSite, mps2::AbstractSite, site::Integer) = update_environments!(env, site, mps1, mps2)
update_environments!(env::Environments, mps::AbstractSite, mpo::AbstractMPOsite, site::Integer) = update_environments!(env, site, mps, mpo, mps)
update_environments!(env::Environments, mps::AbstractSite, site::Integer) = update_environments!(env, site, mps, mps)

# function update_environment!(env::AbstractFiniteEnvironment, mps1::AbstractSite, mpo::ScaledIdentityMPOsite, mps2::AbstractSite, site::Integer)
#     env.L[site+1] = reshape(transfer_matrix(mps1, mpo, mps2, :right)*vec(env.L[site]), size(mps1,3), size(mps2,3))
#     env.R[site-1] = reshape(transfer_matrix(mps1, mpo, mps2, :left)*vec(env.R[site]), size(mps1,1), size(mps2,1))
#     return 
# end

#TODO check performance and compare to matrix multiplication and Tullio
local_mul!(out, envL, envR, mposite::AbstractMPOsite, site::AbstractArray{<:Number,3}) = @tensor out[:] = (envL[-1, 2, 3] * data(mposite)[2, -2, 4, 5]) * (site[3, 4, 1] * envR[-3, 5, 1])

local_mul(envL, envR, mposite::AbstractMPOsite, site::AbstractArray{<:Number,3}) = @tensor temp[:] := (envL[-1, 2, 3] * data(mposite)[2, -2, 4, 5]) * (site[3, 4, 1] * envR[-3, 5, 1])
local_mul(envL, envR, mposite::AbstractMPOsite, site::PhysicalSite) = PhysicalSite(local_mul(envL, envR, mposite, data(site)), ispurification(site))
local_mul(envL, envR, mposite::AbstractMPOsite, site::PVSite) = local_mul(envL, envR, mposite, site.Λ1 * site * site.Λ2)
#TODO implement these for SiteSum
local_mul(envL, envR, site::Array{<:Number,3}) = @tensor temp[:] := envL[-1, 1] * site[1, -2, 2] * envR[-3, 2]
local_mul(envL, envR, site::PhysicalSite) = PhysicalSite(local_mul(envL, envR, data(site)), ispurification(site))
local_mul(envL, envR, site::PVSite) = local_mul(envL, envR, site.Λ1 * site.Γ * site.Λ2)

