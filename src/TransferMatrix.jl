abstract type AbstractTransferMatrix{V} end
struct TransferMatrix{V} <: AbstractTransferMatrix{V}
    op::Any
    ismutating::Bool
end
# struct TransferMatrix!{V} <: AbstractTransferMatrix{V}
#     op::Any
# end

function Base.:*(T::TransferMatrix{V},v::V) where V 
    w::V = ismutating ? T.op(similar(v),v) : T.op(v)
    return w
end
LinearAlgebra.mul!(w::V,T::TransferMatrix{V},v::V) where V = T.op(w,v)::V
# Base.:*(T::TransferMatrix{V},v::V) where V = T.op(similar(v),v)::V

# function Base.:*(T1::TransferMatrix{V},T2::TransferMatrix{V}) where V 
#     if T1ismutating && T2.ismutating
#     TransferMatrix{V}(!ismutating ? (T1.op ∘ T2.op) : (w,v) -> T1.op(w,T2.op(w,v)),)
# end
# Base.:*(Ts::Vector{TransferMatrix{V}},v::V) where V = foldr(*,Ts,init=v)::V
# LinearAlgebra.axpby!
## Structs to organize which transfer function to call 
abstract type AbstractZipLayer{N} end
struct SiteStack{T,N,QN,Sites,Vals} <: AbstractZipLayer{N}
    sites::Sites
    conjs::Vals#NTuple{N,Bool}
    function SiteStack(sites::Tuple,conjs::Vals) where Vals
        #TODO handle sums of sites
        T = eltype(sites[1])
        QN = qntype(sites[1])
        N = length(sites)
        @assert length(sites) == length(conjs)
        @assert length(unique(qntype.(sites))) ==1
        new{T,N,QN,typeof(sites),Vals}(sites,conjs)
    end
end
_typesToTuple(t::Tuple) = Tuple{t...}
qntype(::T) where T = qntype(T)
qntype(::Type{<:AbstractTensor{<:Any,<:Any,QN}}) where {QN} = QN
struct ChainStack{N,MPs,QN,Sites,Vals} <: AbstractVector{SiteStack{N,QN,Sites}}
    chains::MPs
    conjs::Vals#NTuple{N,Bool}
    function ChainStack(chains::Tuple,conjs::Vals) where Vals
        #TODO handle sums of chains
        @assert length(chains) == length(conjs)
        sitetypes = eltype.(chains)
        S = _typesToTuple(sitetypes)
        new{length(chains),typeof(chains),qntype(chains[1]),S,Vals}(chains,conjs)
    end
end
Base.getindex(C::ChainStack,i::Integer) = SiteStack(map(c->c[i],C.chains),C.conjs)

function EnvironmentType(sites::SiteStack{T,N,QN,Sites}) where {N,Sites,QN,T}
    types = EnvironmentType.(sites.sites)
    EnvironmentType(types)
end
_combine_environment_types(::Type{CovariantTensor{T,N,QN}},::Type{CovariantTensor{T,K,QN}}) where {T,N,K,QN} = CovariantTensor{T,N+K,QN}
EnvironmentType(t::NTuple{N,DataType}) where N = foldl(_combine_environment_types,t)
EnvironmentType(::PhysicalSite{T,CovariantTensor{T,3,QN}}) where {T,QN} = CovariantTensor{T,1,QN}
EnvironmentType(::MPOSite{T,CovariantTensor{T,4,QN}},dir) where {T,QN} = CovariantTensor{T,1,QN}

_combine_environment_types(::Type{Array{T,N}},::Type{Array{T,K}}) where {T,N,K} = Array{T,N+K}
EnvironmentType(::PhysicalSite{T,Array{T,3}}) where {T} = Array{T,1}
EnvironmentType(::MPOSite{T,Array{T,4}}) where {T} = Array{T,1}

VirtualSite(::PhysicalSite,dir) = I
function linktransfermatrix(sites::S,dir::Symbol) where S<:SiteStack
    Vs = [VirtualSite(site,reverse_direction(dir)) for site in sites.sites]
    return TransferMatrix{EnvironmentType(sites)}(A->_contract_links(Vs,A))
end
function _contract_links(Vs::Vector{<:VirtualSite},A::AbstractArray{<:Any,N}) where N
    @assert length(Vs) == N
    foldl((A,n)->_contract_single_link(Vs[n].Λ,A,n),1:N,init=A)
end
function _contract_single_link(V::AbstractArray{<:Any,2},A::AbstractArray{<:Any,N},n) where N
    indA = -collect(1:N)
    indA[n] = n
    indV = [n,-n]
    ncon([V,A],[indV,indA])::typeof(A)
end
function _contract_single_link(V::UniformScaling,A::AbstractArray,n) 
    V.λ * A
end

function transfermatrix(ss::SiteStack{T,N,QN,Sites,Vals}) where {T,N,QN,Sites,Vals}
    @assert N>1
    lastc =  1
    endind = sum(ndims.(ss.sites))/2
    function index(k)
        Nsite = ndims(ss.sites[k])
        global lastc,endind
        c = _valFromVal(ss.conjs[k])
        if Nsite == 4
            if k == 1
                out = c ? [-1, 2, endind, 1] : [-1, endind, 2, 1]
            elseif k==N
                out = c ? [-N, endind, lastc, lastc+1] :  [-N, lastc, endind, lastc+1]
                lastc = lastc + 2
            else
                out = c ? [-k, lastc+2, lastc, lastc+1] :  [-k, lastc, lastc+2, lastc+1]
                lastc = lastc + 2
            end
        elseif Nsite == 3
            if k == 1
                out = c ? [-1, 2, 1] :  [-1, endind, 1]
                lastc = c ? 2 : 1
            elseif k==N
                out = c ? [-1, endind, lastc+1] :  [-k, lastc, lastc+1]
            else
                out = c ? [-k, lastc+2, lastc+1] :  [-k, lastc, lastc+1]
                lastc = c  ? lastc+2 : lastc+1
            end
        end
        return out
    end
    zipindices = [index(k) for k in 1:N]
    Rindices = [ind[end] for ind in zipindices]
    function contract(R)
        # index(1) = [-1,last,3,1]
        # index(2) = [-2,3,5,2]
        # index(3) = [-3,5,7,4] # or if last: [-3,5,6,4]
        # index(4) = [-4, 7, 9, 6]
        # index(N) = [-N], , , 

        #indexR = [1, [2k - 2 for k in 2:N]...]
        ncon([R, data.(ss.sites)...], [Rindices,zipindices...],_valFromVal.(ss.vals))
        return R
    end
end
_valFromVal(::Val{T}) where T = T
# _opstring(::Type{Val{false}}) = Symbol("1*")
# _opstring(::Type{Val{true}}) = :conj

# @generated function testgen7(m,vals::Vals) where Vals
#     println(Vals)
#     println(fieldtypes(Vals))
#     ops = _opstring.(fieldtypes(Vals))
#     #m = [1 2im; 3im 4]
#     return :($(ops[1])(m))
# end

# Rtens[r1,r2,...rN]
# ops[n](Tn[-n,un,dn,rn])
# @tensoropt (t1, b1, -1, -2) temp[:] = Rtens[(1:N)...] * conj(Γ[-1, c1, t1]) * Γ[-2, c1, b1]
        

# function linktransfermatrix(V::AbstractVirtualSite,v::AbstractArray{T,N})
#     @tensor 
# end
# Base.:*(T::TransferMatrix{V})

function testTransferMatrix()
    f = v-> cumsum(v,dims=1)
    v = rand(10,10,10)
    T1 = TransferMatrix{typeof(v)}(f)
    tv = T1*v
    ttv = T1*T1*v
    prod(fill(T1,100)) * v
end