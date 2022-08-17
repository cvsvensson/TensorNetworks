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
struct SiteStack{N,Sites} <: AbstractZipLayer{N}
    sites::Sites
    conjs::NTuple{N,Bool}
    function SiteStack(sites::Tuple,conjs)
        #TODO handle sums of sites
        @assert length(sites) == length(conjs)
        new{length(sites),typeof(sites)}(sites,conjs)
    end
end
struct ChainStack{N,MPs,Sites} <: AbstractVector{SiteStack{N,Sites}}
    chains::MPs
    conjs::NTuple{N,Bool}
    function SiteStack(chains::Tuple,conjs)
        #TODO handle sums of chains
        @assert length(sites) == length(conjs)
        new{length(chains),typeof(chains),typeof.(eltype.(chains))}(chains,conjs)
    end
end
Base.getindex(C::ChainStack,i::Integer) = SiteStack(map(c->c[i],C.chains),C.conjs)

function EnvironmentType(sites::SiteStack{N,Sites},dir) where {N,Sites}
    ss = sites.sites
    ft = fieldtypes(Sites)
    s1 = EnvironmentType(ss[1], dir)
    s2 = EnvironmentType(ss[2], dir)
    s3 = EnvironmentType(ss[2], dir)
    s123 = (s1,s2,s3)
    e123 = EnvironmentType(s123...)
    #qns = _QNs.(ss)
    #q = qn(ss[1],:left)
    #qs1 = qn.(ss,:left)
    ts = typeof.(ss)
    #qs = qns(ss)
    types = EnvironmentType.(ss, :left)
    println(types)
    #EnvironmentType(types)
end
function qn(::PhysicalSite{T,CovariantTensor{T,3,Tuple{QNL,QNC,QNR},QN}},dir) where {T,QNL,QNC,QNR,QN}
    if dir == :right 
        return QNR
    else 
        @assert dir ==:left
        return QNL
    end
end
function EnvironmentType(::PhysicalSite{T,CovariantTensor{T,3,Tuple{QNL,QNC,QNR},QN}},dir) where {T,QNL,QNC,QNR,QN}
    if dir == :right 
        return CovariantTensor{T,1,Tuple{QNR},QN}
    else 
        @assert dir ==:left
        return CovariantTensor{T,1,Tuple{QNL},QN}
    end
end
function EnvironmentType(::MPOSite{T,CovariantTensor{T,4,Tuple{QNL,QNU,QND,QNR},QN}},dir) where {T,QNL,QNU,QND,QNR,QN}
    if dir == :right 
        return CovariantTensor{T,1,Tuple{QNR},QN}
    else 
        @assert dir ==:left
        return CovariantTensor{T,1,Tuple{QNL},QN}
    end
end
EnvironmentType(::PhysicalSite{T,Array{T,3}},dir) where {T} = Array{T,1}
EnvironmentType(::MPOSite{T,Array{T,4}},dir) where {T} = Array{T,1}
EnvironmentType(t1::CovariantTensor{T,N,QNs,QN},t2::CovariantTensor{T,1,Tuple{},QN}) where {N,T,QNs,QN} = CovariantTensor{T,N,_QN.(types),QN}
EnvironmentType(types::Vararg{Type{CovariantTensor{T,1,QNs,QN}},N}) where {N,T,QNs,QN} = CovariantTensor{T,N,_QN.(types),QN}
function _combine 
_QNs(::Type{CovariantTensor{<:Any,<:Any,QNs}}) where QNs = QNs
_QN(::Type{CovariantTensor{T,1,QNs,QN}}) where {T,QNs,QN} = fieldtypes(QNs)[1]

VirtualSite(::PhysicalSite,dir) = I
function linktransfermatrix(sites::S,dir::Symbol) where S<:SiteStack
    Vs = [VirtualSite(site,reverse_direction(dir)) for site in sites.sites]
    return TransferMatrix{EnvironmentType(sites,dir)}(A->_contract_links(Vs,A))
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