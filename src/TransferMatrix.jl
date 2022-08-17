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

function linktransfermatrix(sites::SiteStack,dir::Symbol)
    Vs = [VirtualSite(site,:left) for site in sites]

end
function linktransfermatrix(V::AbstractVirtualSite,v::AbstractArray{T,N})
    @tensor 
end
Base.:*(T::TransferMatrix{V})

function testTransferMatrix()
    f = v-> cumsum(v,dims=1)
    v = rand(10,10,10)
    T1 = TransferMatrix{typeof(v)}(f)
    tv = T1*v
    ttv = T1*T1*v
    prod(fill(T1,100)) * v
end