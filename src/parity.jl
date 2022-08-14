using Mods

function _positive_parities(n)
    map(x -> (x >>> -1) + (iseven(count_ones(x)) ? 0 : 1), 0:2^(n-1)-1)
end

const IndexTuple{N} = NTuple{N,Int}
abstract type AbstractQuantumNumber end
#const QNTuple{N,QN} = NTuple{N,QN<:AbstractQuantumNumber}

#const Link{QN} = Tuple{QN,Bool}
#const LinkTuple{QNs} = NTuple{N,Link{}}
#const LinkTuple{N} = NTuple{N,<:AbstractQuantumNumber}

struct IdentityQuantumNumber <: AbstractQuantumNumber end
# struct ParityQN <: AbstractQuantumNumber
#     parity::Bool
# end
struct ZQuantumNumber{N} <: AbstractQuantumNumber
    n::Mod{N,Int64}
end
Base.:*(a::Number, b::Z) where {Z<:ZQuantumNumber} = ZQuantumNumber(a * b.n)
Base.:*(a::Z, b::Number) where {Z<:ZQuantumNumber} = ZQuantumNumber(a.n * b)
const ParityQN = ZQuantumNumber{2}
struct U1QuantumNumber{T} <: AbstractQuantumNumber
    n::T
end

struct CovariantTensor{T,N,QNs<:Tuple,QN<:AbstractQuantumNumber} <: AbstractArray{T,N}
    blocks::Vector{Array{T,N}}
    qns::Vector{QNs}
    dirs::NTuple{N,Bool}
    qntotal::QN
    function CovariantTensor(blocks::Vector{Array{T,N}}, qns::Vector{QNs}, dirs::NTuple{N,Bool}, qntotal::QN) where {QN,QNs,N,T}
        @assert length(blocks) == length(qns)
        new{T,N,QNs,QN}(blocks, qns, dirs, qntotal)
    end
end
Base.size(A::CovariantTensor) = foldl(.*, size.(A.blocks))
Base.show(io::IO, A::CovariantTensor{T,N,QNs,QN}) where {T,N,QNs,QN} = println(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)
Base.show(io::IO, ::MIME"text/plain", A::CovariantTensor{T,N,QNs,QN}) where {T,N,QNs,QN} = print(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)

get_elements(t::Tuple, p) = map(i -> t[i], p)

# function similarstructure_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A::CovariantTensor, CA::Symbol)
#     p = (p1..., p2...)
#     permutef(t) = permute(t, p)
#     qns = permutef.(A.qns)
#     qntotal = A.qntotal
#     dirs = permutef(A.dirs)
#     sizes = permutef.(size.(A.blocks))
#     if CA == :N
#         return (T, sizes, qns, dirs, qntotal)
#     else
#         @assert CA == :C
#         return (T, sizes, qns, map(!, dirs), invert(qntotal))
#     end
# end
function TensorOperations.similarstructure_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A::CovariantTensor, CA::Symbol=:N)
    op = CA==:N ? identity : invert
    ind = (p1..., p2...)
    sizes = [map(n -> size(block, n), ind) for block in A.blocks]
    dirs = map(n -> A.dirs[n],ind) 
    qns = [map(n -> op(qns[n]),ind) for qns in A.qns]
    (sizes, qns, dirs, op(A.qntotal))
end


#  _similarstructure_from_indices(T, (p1..., p2...), A)

# function _similarstructure_from_indices(T, ind, A::CovariantTensor)
#     sizes = [map(n -> size(block, n), ind) for block in A.blocks]
#     qns = [qns[ind] for qns in A.qns]
#     dirs = A.dirs[ind]
#     (sizes, qns, dirs, A.qntotal)
# end

function TensorOperations.similarstructure_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple,
    A::CovariantTensor, B::CovariantTensor,
    CA::Symbol=:N, CB::Symbol=:N)
    indC = (p1..., p2...)

    #_similarstructure_from_indices(T, poA, poB, (p1..., p2...), A, B)
end
function tensorcontract(A::CovariantTensor, IA::Tuple, B::CovariantTensor, IB::Tuple, IC::Tuple)
    oindA, cindA, oindB, cindB, indCinoAB = contract_indices(IA, IB, IC)

    T = promote_type(eltype(A), eltype(B))
    pairs = find_matches(A, cIA, B, cIB)
    oqnA = [A.qns[nA] for (nA, nB) in pairs]
    oqnB = [B.qns[nB] for (nA, nB) in pairs]
    oqnC = [get_elements((A.qns[nA]..., B.qns[nB]...), indCinoAB) for (nA, nB) in pairs]

    C = similar_from_indices(T, oindA, oindB, indCinoAB, (), A, B, :N, :N)

    contract!(1, A, :N, B, :N, 0, C, oindA, cindA, oindB, cindB, indCinoAB)
    return C
end

function _similarstructure_from_indices(T, poA::IndexTuple, poB::IndexTuple,
    ind::IndexTuple, A::AbstractArray, B::AbstractArray)
    oindA, cindA, oindB, cindB, indCinoAB = TensorOperations.contract_indices(IA, IB, IC)
    pairs = find_matches(A,)

    oszA = map(n -> size(A, n), poA)
    oszB = map(n -> size(B, n), poB)
    sz = let osz = (oszA..., oszB...)
        map(n -> osz[n], ind)
    end
    return sz
end

# function similar_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A, CA::Symbol)
#     structure = similarstructure_from_indices(T, p1, p2, A, CA)
#     similar_from_structure(A, T, structure)
# end
# similar_from_indices(eltype(A), indCinA, (), A, :N)

function Base.similar(A::CovariantTensor{T}, type::Type=T) where {T,N}
    Tnew = promote_type(T,type)
    blocks = similar.(A.blocks, Tnew)
    qns = deepcopy(A.qns)
    dirs = deepcopy(A.dirs)
    qntotal = deepcopy(A.qntotal)
    CovariantTensor(blocks, qns, dirs, qntotal)
end
const CovStructure{N,QNs,QN} = Tuple{Vector{NTuple{N,Int}},Vector{QNs},NTuple{N,Bool},QN}
function structure(A::CovariantTensor)
    sizes = size.(A.blocks)
    return (sizes,A.qns,A.dirs,Q.qntotal)
end
function Base.similar(A::CovariantTensor{T}, type::Type=T, (sizes, qns, dirs, qntotal)::CovStructure{N} = structure(A)) where {T,N}
    Tnew = promote_type(T,type)
    blocks = [Array{Tnew,N}(undef,sz) for sz in sizes]
    CovariantTensor(blocks, qns, dirs, qntotal)
end

# function similar_from_structure(A::CovariantTensor, T, structure)
#     blocks = similar.(A.blocks,T)
#     qns = deepcopy(A.qns)
#     dirs = deepcopy(A.dirs)
#     qntotal = deepcopy(A.qntotal)
# end

function Base.:*(a::Number, A::CovariantTensor)
    B = similar(A)
    Threads.@threads for n in eachindex(B.blocks)
        B.blocks[n] = a * A.blocks[n]
    end
    return B
end
function LinearAlgebra.lmul!(a::Number, B::CovariantTensor)
    Threads.@threads for n in eachindex(B.blocks)
        LinearAlgebra.lmul!(a, B.blocks[n])
    end
    return B
end

#links(A::CovariantTensor) = map(tuple, A.qns, A.dirs)
# struct QNTuple{QNs}
#     qns::QNs
#     function QNTuple(qns::QNs) where {QNs <: Tuple}
#         @assert all(map(qn->typeof(qn),qns))
#         new{QNs}(qns)
#     end
# end
function SymmetricTensor(blocks::Vector, qns, dirs)
    @assert all(map(qn -> iszero(fuse(qn, dirs)), qns)) "$qns \n $dirs \n $(map(qn->fuse(qn,dirs),qns) )"
    CovariantTensor(blocks, qns, dirs, IdentityQuantumNumber())
end


#QuantumNumber(l::Link) = l[2] ? invert(l[1]) : l[1]
==(a::ZQuantumNumber{N}, b::ZQuantumNumber{N}) where N = a.n == b.n
isequal(a::ZQuantumNumber{N}, b::ZQuantumNumber{N}) where N= isequal(a.n,b.n)
Base.hash(a::ZQuantumNumber, h::UInt64 = UInt64(0)) = Base.hash(a.n, h)
#Base.:!(qn::ParityQN) = ParityQN(!qn.parity)
#fuse(qn1::ParityQN,qn2::ParityQN) = qn1.parity + qn2.parity
invert(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
#invert(l::ParityQN) = !l
#invert(ls::QNTuple) = QNTuple(invert.(ls.qns))
Base.:+(lA::ZQuantumNumber{N}, lB::ZQuantumNumber{N}) where {N} = ZQuantumNumber{N}(lA.n + lB.n)
Base.:-(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
Base.iszero(qn::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = iszero(qn.n)
fuse(l1::QN, l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = l1 + l2
fuse(l1, l2::IdentityQuantumNumber) = l1
fuse(l1::IdentityQuantumNumber, l2) = l2
#fuse(l1::QNs, l2::QNs) where {QNs<:Tuple} = map(fuse, l1, l2)
#fuse(l1::Link{QN},l2::Link{QN}) where {QN<:AbstractQuantumNumber}= fuse(QuantumNumber(l1),QuantumNumber(l2))
#fuse(ls::NTuple{<:Any,Union{Link,AbstractQuantumNumber}}) = foldl(fuse,ls)
fuse(l::AbstractQuantumNumber, d::Bool) = d ? invert(l) : l
fuse(l1s::NTuple{N}, d1s::NTuple{N,Bool}) where {N} = fuse(map(fuse, l1s, d1s))
fuse(l1s::Tuple) = foldl(fuse, l1s)
fuse(l1s::Vector{QN}) where {QN<:AbstractQuantumNumber} = foldl(fuse, l1s)
fuse(l1s::Vector{QN}, d1s::NTuple{N,Bool}) where {N,QN<:AbstractQuantumNumber} = fuse(tuple(l1s...), d1s)
#fuse(l1s::QNTuple{N1}, d1s::NTuple{N1,Bool}, l2s::QNTuple{N2}, d2s::NTuple{N2,Bool}) where {N1,N2} = fuse((l1s..., l2s...), (d1s..., d2s...))

function find_matches(A::CovariantTensor, IA::NTuple{NA,Int}, B::CovariantTensor, IB::NTuple{NB,Int}) where {NA,NB}
    pairs = NTuple{2,Int}[]
    for (n1, l1) in enumerate(links(A))
        for (n2, l2) in enumerate(links(B))
            if l1.qn[I1] == l2.qn[I2] && xor(l1.dir, l2.dir)
                append!(pairs, (n1, n2))
            end
        end
    end
end
function tensorcontract(A::CovariantTensor{T1}, IA::Tuple, B::CovariantTensor{T2}, IB::Tuple) where {T1,T2}
    qntotal = fuse(A.qntotal, B.qntotal)
    #T = promote_type(T1,T2)
    pairs = find_matches(A, IA, B, IB)

    blocks = [tensorcontract(A.blocks[n1], IA, B.blocks[n2], IB) for (n1, n2) in pairs]
    qns = [tuple(A.qns[n1], IA, B.qns[n2], IB) for (n1, n2) in pairs]
end

#directedQN(A) = map(tuple,A.qns,A.dirs)
"""
    add!(α, A, conjA, β, C, indleft, indright)
Implements `C = β*C+α*permute(op(A))` where `A` is permuted such that the left (right)
indices of `C` correspond to the indices `indleft` (`indright`) of `A`, and `op` is `conj`
if `conjA == :C` or the identity map if `conjA == :N` (default). Together,
`(indleft..., indright...)` is a permutation of 1 to the number of indices (dimensions) of
`A`.
"""
function TensorOperations.add!(α, A::CovariantTensor, CA::Symbol, β, C::CovariantTensor, indleft::IndexTuple,
    indright::IndexTuple)
    indCinA = (indleft..., indright...)
    Aqnp = [get_elements(l, indCinA) for l in A.qns]
    Adirp = get_elements(A.dirs, indCinA)
    @assert Aqnp == C.qns "qns don't match in add! $Aqnp, $(C.qns)"
    @assert Adirp == C.dirs "dirs don't match in add!"
    for n in eachindex(A.blocks)
        TensorOperations.add!(α, A.blocks[n], CA, β, C.blocks[n], indleft, indright)
    end
    return C
end

function tensortrace_test(A, IA::Tuple, IC::Tuple)
    indCinA, cindA1, cindA2 = TensorOperations.trace_indices(IA, IC)
    C = TensorOperations.similar_from_indices(eltype(A), indCinA, (), A, :N)
    #TensorOperations.trace!(1, A, :N, 0, C, indCinA, cindA1, cindA2)
    return C
end

"""
    trace!(α, A, conjA, β, C, indleft, indright, cind1, cind2)
Implements `C = β*C+α*partialtrace(op(A))` where `A` is permuted and partially traced,
such that the left (right) indices of `C` correspond to the indices `indleft` (`indright`)
of `A`, and indices `cindA1` are contracted with indices `cindA2`. Furthermore, `op` is
`conj` if `conjA == :C` or the identity map if `conjA=:N` (default). Together,
`(indleft..., indright..., cind1, cind2)` is a permutation of 1 to the number of indices
(dimensions) of `A`.
"""
function TensorOperations.trace!(α, A::CovariantTensor, CA::Symbol, β, C::CovariantTensor, indleft::IndexTuple,
        indright::IndexTuple, cind1::IndexTuple, cind2::IndexTuple)
    #indCinA = (indleft...,indright...)
    @assert all(map(xor, map(n->A.dirs[n],cind1),  map(n->A.dirs[n],cind2)))
    match_indices = find_matches(A,cind1,cind2)
    nmi = length(match_indices)
    nc = length(C.blocks)
    if nc !== nmi
        @warn  "Pre-allocation in trace! is of the wrong length: $nc !== $nmi" 
    end
    for (Cblock,n) in zip(C.blocks,match_indices)
        TensorOperations.trace!(α,A.blocks[n],CA,β,Cblock,indleft,indright,cind1,cind2)
    end
    #deleteat!(C.blocks,nmi+1:nc)
    cat_collisions!(C)
    C
    #@assert all(iszero.(fuse(A.qns)))
    #all(map(fuse, map(n->qn[n],cind1),  map(n->qn[n],cind2)))
end
function find_matches(A::CovariantTensor,cind1::IndexTuple,cind2::IndexTuple)
    return [n for (n,qn) in enumerate(A.qns) if get_elements(qn,cind1) == get_elements(qn,cind2)]
end
# function TensorOperations.add!(α, A::CovariantTensor, CA::Symbol, β, C::CovariantTensor, indleft::IndexTuple,
#     indright::IndexTuple)
#     indCinA = (indleft..., indright...)
#     Aqnp = [map(i->l[i],indCinA) for l in A.qns]
#     Adirp = [map(i->l[i],indCinA) for l in A.dirs]
#     @assert Aqnp == C.qns "qns don't match in add!"
#     @assert Adirp == C.dirs "dirs don't match in add!"
#     Threads.@threads for n in eachindex(A.blocks)
#         TensorOperations.add!(α,A.blocks[n],CA,β,C.blocks[n],indleft,indright)
#     end
#     return C
# end
function _rand_qn(::Val{N}, dirs::NTuple{Nd,Bool}) where {N,Nd}
    qns0 = [ZQuantumNumber{N}(rand(Int)) for k in 1:Nd-1]
    push!(qns0, ZQuantumNumber{N}(0))
    op = dirs[end] ? identity : -
    qns::NTuple{Nd,ZQuantumNumber{N}} = (qns0[begin:end-1]..., op(fuse(qns0, dirs)))
end
function _id(n, m)
    dirs = tuple(rand(Bool, 2)...)
    qns = [_rand_qn(Val(2), dirs) for k in 1:m]
    # qns = [(op = rand(Bool) ? identity : !; map(ParityQN ∘ Int ∘ op,dirs)) for k in 1:m]
    blocks = [Array(1.0I, n, n) for k in 1:m]
    return SymmetricTensor(blocks, qns, dirs)
end
function _test_add(n, m, a=1, b=0)
    A = _id(n, m)
    C = 0 * A
    Afull = Array(1.0I, n * m, n * m)
    Cfull = 0.0 * Array(1.0I, n * m, n * m)
    println("add!")
    @time TensorOperations.add!(a, A, :N, b, C, (1, 2), ())
    println("@tensor =")
    @time @tensor C[:] += a * A[-1, -2]
    println("@tensor :=")
    @time @tensor C2[:] := a * A[-1, -2]
    println("Dense")
    @time Cfull .= a * Afull + b * Cfull

    @tensor out[:] := A[1,1]
    out
end

function cat_collisions(A::CovariantTensor)
    reduced_qns = unique(A.qns)
    indA = [findall(qna->qn==qna,A.qns) for qn in reduced_qns]
    println(indA)
    reduced_blocks = [block_diagonal(getindex(A.blocks,inds)) for inds in indA]
    return CovariantTensor(reduced_blocks,reduced_qns,A.dirs,A.qntotal)
end

function cat_collisions!(A::CovariantTensor)
    reduced_qns = unique(A.qns)
    indA = [findall(qna->qn==qna,A.qns) for qn in reduced_qns]
    println(indA[1])
    reduced_blocks = [block_diagonal(getindex(A.blocks,inds)) for inds in indA]
    replace_vector!(A.blocks,reduced_blocks)
    replace_vector!(A.qns,reduced_qns)
    return A
end

function block_diagonal(blocks::Vector{Array{T,N}}) where {T,N}
    sizes = size.(blocks)
    fullsize = reduce(.+,sizes)
    A::Array{T,N} = zeros(T,fullsize)
    previndices = fill(0,N)
    for n in eachindex(blocks)
        nextstart = 1 .+ previndices
        nextstop = previndices .+ sizes[n]
        ranges = map(:,nextstart,nextstop)
        A[ranges...] .= blocks[n]
        previndices=nextstop
    end
    return A
end

function replace_vector!(list1::Vector,list2::Vector)
    deleteat!(list1,eachindex(list1))
    append!(list1,list2)
end