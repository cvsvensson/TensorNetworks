using Mods

function _positive_parities(n)
    map(x -> (x >>> -1) + (iseven(count_ones(x)) ? 0 : 1), 0:2^(n-1)-1)
end

const IndexTuple{N} = NTuple{N,Int}
abstract type AbstractQuantumNumber end

struct IdentityQN <: AbstractQuantumNumber end

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
        @assert length(qns[1]) == N
        @assert all(map(qn->iszero(fuse(fuse(qn),qntotal)), qns))
        A = new{T,N,QNs,QN}(blocks, qns, dirs, qntotal)
        return A
    end
end
Base.size(A::CovariantTensor) = foldl(.+, size.(A.blocks))
Base.show(io::IO, A::CovariantTensor{T,N,QNs,QN}) where {T,N,QNs,QN} = println(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)
Base.show(io::IO, ::MIME"text/plain", A::CovariantTensor{T,N,QNs,QN}) where {T,N,QNs,QN} = print(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)

getelements(t::Tuple, p) = map(i -> t[i], p)

function TensorOperations.similarstructure_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A::CovariantTensor, CA::Symbol=:N)
    op = CA==:N ? identity : invert
    ind = (p1..., p2...)
    sizes = [map(n -> size(block, n), ind) for block in A.blocks]
    dirs = op.(getelements(A.dirs,ind))
    qns = [getelements(qns,ind) for qns in A.qns]
    (sizes, qns, dirs, A.qntotal)
end

function TensorOperations.similarstructure_from_indices(T::Type, poA::IndexTuple, poB::IndexTuple,
    p1::IndexTuple, p2::IndexTuple,
    A::CovariantTensor{<:Any,NA}, B::CovariantTensor{<:Any,NB},
    CA::Symbol=:N, CB::Symbol=:N) where {NA,NB}
    indC = (p1..., p2...)
    opA = CA==:N ? identity : invert
    opB = CB==:N ? identity : invert
    dirsA = opA.(getelements(A.dirs,poA))
    dirsB =  opB.(getelements(B.dirs,poB))
    sizes = vec([getelements((map(n->size(A.blocks[nA],n),poA)...,map(n->size(B.blocks[nB],n),poB)...),indC) for nA in eachindex(A.blocks), nB in eachindex(B.blocks)])
    dirs = getelements((dirsA...,dirsB...),indC)
    qns = vec([getelements((getelements(A.qns[nA],poA)...,getelements(B.qns[nB],poB)...),indC) for nA in eachindex(A.blocks), nB in eachindex(B.blocks)])
    qntotal = fuse(A.qntotal,B.qntotal)
    correct_inds = [iszero(fuse(fuse(qn),qntotal)) for qn in qns]
    (sizes[correct_inds], qns[correct_inds], dirs, qntotal)
    #println((sizes, qns, dirs, fuse(A.qntotal,B.qntotal)))
    #println(cat_collisions(s))
    #cat_collisions(s)
end

# function Base.similar(A::CovariantTensor{T}, type::Type=T) where {T,N}
#     Tnew = promote_type(T,type)
#     blocks = similar.(A.blocks, Tnew)
#     qns = deepcopy(A.qns)
#     dirs = deepcopy(A.dirs)
#     qntotal = deepcopy(A.qntotal)
#     CovariantTensor(blocks, qns, dirs, qntotal)
# end
const CovStructure{N,QNs,QN} = Tuple{Vector{NTuple{N,Int}},Vector{QNs},NTuple{N,Bool},QN}
function structure(A::CovariantTensor)
    sizes = size.(A.blocks)
    return (sizes,A.qns,A.dirs,A.qntotal)
end

function Base.similar(A::CovariantTensor{T}, (sizes, qns, dirs, qntotal)::CovStructure{N} = structure(A)) where {T,N}
    blocks = [zeros(T,sz) for sz in sizes]
    CovariantTensor(blocks, qns, dirs, qntotal)
end
function Base.similar(A::CovariantTensor, type::Type, (sizes, qns, dirs, qntotal)::CovStructure{N} = structure(A)) where {N}
    blocks::Vector{Array{type,N}} = [zeros(type,sz) for sz in sizes]
    CovariantTensor(blocks, qns, dirs, qntotal)
end

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

function SymmetricTensor(blocks::Vector, qns, dirs)
    @assert all(map(qn -> iszero(fuse(qn, dirs)), qns)) "$qns \n $dirs \n $(map(qn->fuse(qn,dirs),qns) )"
    CovariantTensor(blocks, qns, dirs, IdentityQN())
end


#QuantumNumber(l::Link) = l[2] ? invert(l[1]) : l[1]
Base.:(==)(a::ZQuantumNumber{N}, b::ZQuantumNumber{N}) where N = a.n == b.n
Base.hash(a::ZQuantumNumber, h::UInt64 = UInt64(0)) = Base.hash(a.n, h)
#Base.:!(qn::ParityQN) = ParityQN(!qn.parity)
#fuse(qn1::ParityQN,qn2::ParityQN) = qn1.parity + qn2.parity
invert(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
#invert(l::ParityQN) = !l
#invert(ls::QNTuple) = QNTuple(invert.(ls.qns))
# Base.:+(lA::ZQuantumNumber{N}, lB::ZQuantumNumber{N}) where {N} = ZQuantumNumber{N}(lA.n + lB.n)
Base.:-(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
Base.:-(l::QN,l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(l.n - l2.n)
Base.:+(l::QN,l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(l.n + l2.n)
Base.iszero(qn::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = iszero(qn.n)
Base.iszero(qn::IdentityQN) = true
fuse(l1::QN, l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = l1 + l2
fuse(l1, l2::IdentityQN) = l1
fuse(l1::IdentityQN, l2) = l2
fuse(l1::IdentityQN, l2::IdentityQN) = IdentityQN()
#fuse(l1::QNs, l2::QNs) where {QNs<:Tuple} = map(fuse, l1, l2)
#fuse(l1::Link{QN},l2::Link{QN}) where {QN<:AbstractQuantumNumber}= fuse(QuantumNumber(l1),QuantumNumber(l2))
#fuse(ls::NTuple{<:Any,Union{Link,AbstractQuantumNumber}}) = foldl(fuse,ls)
fuse(l::AbstractQuantumNumber, d::Bool) = d ? invert(l) : l
fuse(l1s::NTuple{N}, d1s::NTuple{N,Bool}) where {N} = fuse(map(fuse, l1s, d1s))
fuse(l1s::Tuple) = foldl(fuse, l1s)
fuse(l1s::Tuple{}) = IdentityQN()
fuse(l1s::Vector{QN}) where {QN<:AbstractQuantumNumber} = foldl(fuse, l1s)
fuse(l1s::Vector{QN}, d1s::NTuple{N,Bool}) where {N,QN<:AbstractQuantumNumber} = fuse(tuple(l1s...), d1s)
#fuse(l1s::QNTuple{N1}, d1s::NTuple{N1,Bool}, l2s::QNTuple{N2}, d2s::NTuple{N2,Bool}) where {N1,N2} = fuse((l1s..., l2s...), (d1s..., d2s...))


function TensorOperations.add!(α, A::CovariantTensor, CA::Symbol, β, C::CovariantTensor, indleft::IndexTuple,
    indright::IndexTuple)
    indCinA = (indleft..., indright...)
    Aqnp = [getelements(l, indCinA) for l in A.qns]
    Adirp = getelements(A.dirs, indCinA)
    opA = CA ==:N ? identity : invert
    @assert Aqnp == C.qns "qns don't match in add! \n $Aqnp,\n $(C.qns)"
    @assert opA.(Adirp) == C.dirs "dirs don't match in add!"
    for n in eachindex(A.blocks)
        TensorOperations.add!(α, A.blocks[n], CA, β, C.blocks[n], indleft, indright)
    end
    remove_zeros!(C)
    cat_collisions!(C)
    return C
end


function TensorOperations.trace!(α, A::CovariantTensor, CA::Symbol, β, C::CovariantTensor, indleft::IndexTuple,
        indright::IndexTuple, cind1::IndexTuple, cind2::IndexTuple)
    #indCinA = (indleft...,indright...)
    @assert all(map(xor, getelements(A.dirs,cind1), getelements(A.dirs,cind2)))
    match_indices = find_matches(A,cind1,cind2)
    nmi = length(match_indices)
    nc = length(C.blocks)
    if nc !== nmi
        @warn  "Pre-allocation in trace! is of the wrong length: $nc !== $nmi" 
    end
    for (Cblock,n) in zip(C.blocks,match_indices)
        TensorOperations.trace!(α,A.blocks[n],CA,β,Cblock,indleft,indright,cind1,cind2)
    end
    remove_zeros!(C)
    cat_collisions!(C)
    return C
end
function find_matches(A::CovariantTensor,cind1::IndexTuple,cind2::IndexTuple)
    pairs = [n for (n,qn) in enumerate(A.qns) if getelements(qn,cind1) == getelements(qn,cind2)]
end

function TensorOperations.contract!(α, A::CovariantTensor, CA::Symbol, B::CovariantTensor, CB::Symbol,
        β, C::CovariantTensor, oindA::IndexTuple, cindA::IndexTuple, oindB::IndexTuple,
        cindB::IndexTuple, indleft::IndexTuple, indright::IndexTuple, syms = nothing) 
        opA = CA ==:N ? identity : invert
        opB = CB ==:N ? identity : invert
        @assert all(map(xor, opA.(getelements(A.dirs,cindA)), opB.(getelements(B.dirs,cindB))))
        @assert length(unique(A.qns))==length(A.qns) && length(unique(B.qns))==length(B.qns)
        pairs = find_matches(A,cindA,B,cindB)
        Cind = LinearIndices((length(A.blocks),length(B.blocks)))
        for (nC,(nA,nB)) in enumerate(pairs)
            #C.blocks[Cind[nA,nB]]
            TensorOperations.contract!(α,A.blocks[nA],CA,B.blocks[nB],CB,β,C.blocks[nC],oindA,cindA,oindB,cindB,indleft,indright)
        end
        remove_zeros!(C)
        cat_collisions!(C)
        return C
end
invert(b::Bool) = !b
function find_matches(A::CovariantTensor, cindA::IndexTuple, B::CovariantTensor, cindB::IndexTuple)
    pairs = [(nA, nB) for nA in eachindex(A.qns), nB in eachindex(B.qns) if getelements(A.qns[nA],cindA) == getelements(B.qns[nB],cindB)]
end

function _rand_qn(::Val{N}, dirs::NTuple{Nd,Bool}) where {N,Nd}
    qns0 = [ZQuantumNumber{N}(rand(Int)) for k in 1:Nd-1]
    push!(qns0, ZQuantumNumber{N}(0))
    # op = dirs[end] ? identity : -
    qns::NTuple{Nd,ZQuantumNumber{N}} = (qns0[begin:end-1]..., fuse(fuse(qns0, dirs),dirs[end]))
end
function _id(n, m)
    dirs = (true,false)#tuple(rand(Bool, 2)...)
    qns = [_rand_qn(Val(2), dirs) for k in 1:m]
    # qns = [(op = rand(Bool) ? identity : !; map(ParityQN ∘ Int ∘ op,dirs)) for k in 1:m]
    blocks = [Array(1.0I, n, n) for k in 1:m]
    return SymmetricTensor(blocks, qns, dirs)
end
function _test_add(n, m, a=1, b=0)
    A = _id(n, m)
    C1 = 0*deepcopy(A)
    C2 = 0*deepcopy(A)
    Afull = Array(1.0I, n * m, n * m)
    Cfull = 0.0 * Array(1.0I, n * m, n * m)
    println("add!")
    @time TensorOperations.add!(a, A, :N, b, C1, (1, 2), ())
    println("@tensor =")
    @time @tensor C2[:] += a * A[-1, -2]
    println("@tensor :=")
    @time @tensor C3[:] := a * A[-1, -2]
    println("Dense")
    @time Cfull .= a * Afull + b * Cfull
    println("Trace")
    @tensor out[:] := A[1,1]
    println("Contract")
    @tensor Cc[:] := A[-1,2]*A[2,-2]
    Cc
end

function cat_collisions(A::CovariantTensor)
    reduced_qns = unique(A.qns)
    indA = [findall(qna->qn==qna,A.qns) for qn in reduced_qns]
    reduced_blocks = [block_diagonal(getindex(A.blocks,inds)) for inds in indA]
    return CovariantTensor(reduced_blocks,reduced_qns,A.dirs,A.qntotal)
end

function cat_collisions!(A::CovariantTensor)
    reduced_qns = unique(A.qns)
    indA = [findall(qna->qn==qna,A.qns) for qn in reduced_qns]
    reduced_blocks = [block_diagonal(getindex(A.blocks,inds)) for inds in indA]
    replace_vector!(A.blocks,reduced_blocks)
    replace_vector!(A.qns,reduced_qns)
    return A
end
function cat_collisions(s::CovStructure)
    (sizes,qns,dirs,qntotal) = s
    reduced_qns = unique(qns)
    indices = [findall(qna->qn==qna,qns) for qn in reduced_qns]
    reduced_sizes = [reduce(.+,getindex(sizes,inds)) for inds in indices]
    return (reduced_sizes,reduced_qns,dirs,qntotal)
end

function remove_zeros!(A::CovariantTensor)
    inds =  norm.(A.blocks) .== 0
    deleteat!(A.blocks,inds)
    deleteat!(A.qns,inds)
end
function block_diagonal(blocks::Vector{Array{T,0}}) where {T}
    sum(blocks)
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
TensorOperations.scalar(C::CovariantTensor) = ndims(C)==0 && C.qntotal == IdentityQN() ? sum(C.blocks)[] : throw(DimensionMismatch())
TensorOperations.memsize(a::CovariantTensor) = Base.summarysize(a)

Base.Array(A::CovariantTensor) = block_diagonal(A.blocks)

function Base.rand(qntuple::Type{<:NTuple{N,<:AbstractQuantumNumber}}, qntotal::AbstractQuantumNumber = IdentityQN()) where N
    qntypes = fieldtypes(qntuple)
    qns = [rand(qn) for qn in qntypes]
    qns[end] = -fuse(qns[begin:end-1])
    qns[end] = fuse(qns[end],qntotal)
    Tuple(qns)
end

function Base.rand(::Type{CovariantTensor{T,N,QNs,QN}}, n::Integer = 0) where {T,N,QNs,QN}
    n = rand(1:10)
    sizes = rand(1:10,n,N)
    blocks = [rand(T,sizes[k,:]...) for k in 1:n]
    dirs = Tuple(rand(Bool,N))
    QNstypes = fieldtypes(QNs)
    qns = [(qns = [rand(qn) for qn in QNstypes]; qns[end] -= fuse(qns); Tuple(qns)) for k in 1:n]
    qntotal = rand(QN)
    A = CovariantTensor(blocks,qns,dirs,qntotal)::CovariantTensor{T,N,QNs,QN}
    cat_collisions!(A)
end
Base.rand(::Type{IdentityQN}) = IdentityQN()
Base.rand(::Type{ZQuantumNumber{N}}) where {N} = ZQuantumNumber{N}(rand(Mod{N}))

function rand_compatible_tensor(A::CovariantTensor{T,N,QNs,QN}) where {T,N,QNs,QN}
    n = length(A.blocks)
    sizes = size.(A.blocks)
    blocks = [rand(T,sizes[k,:]...) for k in 1:n]
    return CovariantTensor(blocks,A.qns,invert.(A.dirs),A.qntotal)#::CovariantTensor{T,N,QNs,QN}
end

