
using Mods

function _positive_parities(n)
    map(x-> (x >>> -1) + (iseven(count_ones(x)) ? 0 : 1), 0:2^(n-1)-1)
end

abstract type AbstractQuantumNumber end

struct CovariantTensor{T,N,QNs<:Tuple,QN<:AbstractQuantumNumber} <: AbstractArray{T,N}
    blocks::Vector{Array{T,N}}
    qns::Vector{Link{QN}}
    # dirs::Vector{Bool}
    qntotal::Link{QN}
    function CovariantTensor(blocks::Vector{Array{N,T}}, qns::Vector{Link{QNs}},qntotal::Link{QN}) where {QN,QNs}
        @assert length(blocks)==length(qns)==length(dirs)
        new{T,N,QNs,QN}(blocks,qns,qntotal)
    end
end
struct IdentityQuantumNumber <: AbstractQuantumNumber end
# struct QNTuple{QNs}
#     qns::QNs
#     function QNTuple(qns::QNs) where {QNs <: Tuple}
#         @assert all(map(qn->typeof(qn),qns))
#         new{QNs}(qns)
#     end
# end
function SymmetricTensor(blocks::Vector{Array{N,T}},qns::Vector{QNs},dirs) where {QNs}
    CovariantTensor(blocks,qns,dirs,Link(IdentityQuantumNumber(),false))
end
struct Link{QN}
    qn::QN
    dir::Bool
end

struct ParityQN <: AbstractQuantumNumber
    parity::Bool
end
struct ZQuantumNumber{N} <: AbstractQuantumNumber
    n::Mod{N}
end

QuantumNumber(l::Link{QN}) = l.dir ? invert(l.qn) : qn

invert(l::Union{ZQuantumNumber,U1QuantumNumber}) = -l
invert(ls::QNTuple) = QNTuple(invert.(ls.qns))
fuse(l1::QN,l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = l1 + l2
fuse(l1,l2::IdentityQuantumNumber) = l1
fuse(l1::IdentityQuantumNumber,l2) = l2
fuse(l1::QNs,l2::QNs) where {QNs<:QNTuple} = map(fuse,l1,l2)
fuse(l1::Link{QN},l2::Link{QN}) where {QN<:AbstractQuantumNumber}= fuse(QuantumNumber(l1),QuantumNumber(l2))
fuse(ls::NTuple{<:Any,Union{Link,AbstractQuantumNumber}}) = foldl(fuse,ls)


function tensorcontract(t1::CovariantTensor{T1,N1},I1,t2::CovariantTensor{T2,N2},I2) where {T1,T2,N1,N2}
    qntotal = fuse(t1.qntotal,t2.qntotal)
    T = promote_type(T1,T2)
    pairs = Tuple{Int,Int}[]
    for (n1,l1) in enumerate(links(t1))
        for (n2,l2) in enumerate(links(t2))
            if l1.qn[I1] == l2.qn[I2] && xor(l1.dir,l2.dir)
                append!(pairs,(n1,n2))
            end
        end
    end
    blocks = Vector{Array{T,N1+}}
    blocks = [tensorcontract(t1.blocks[n1],I1,t2.blocks[n2],I2) for (n1,n2) in pairs] 

end

