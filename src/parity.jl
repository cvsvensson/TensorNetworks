function _positive_parities(n)
    map(x -> (x >>> -1) + (iseven(count_ones(x)) ? 0 : 1), 0:2^(n-1)-1)
end

Base.:*(a::Number, b::Z) where {Z<:ZQuantumNumber} = ZQuantumNumber(a * b.n)
Base.:*(a::Z, b::Number) where {Z<:ZQuantumNumber} = ZQuantumNumber(a.n * b)

Base.size(A::CovariantTensor) = foldl(.+, size.(A.blocks))
Base.show(io::IO, A::CovariantTensor{T,N,QN}) where {T,N,QN} = println(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)
Base.show(io::IO, ::MIME"text/plain", A::CovariantTensor{T,N,QN}) where {T,N,QN} = print(io, "CovariantTensor{", T, ",", N, "}\n", A.qns, "\n", A.dirs, "\n", A.qntotal)

getelements(t::Tuple, p) = map(i -> t[i], p)

function TensorOperations.similarstructure_from_indices(T::Type, p1::IndexTuple, p2::IndexTuple, A::CovariantTensor, CA::Symbol=:N)
    op = CA==:N ? identity : invert
    ind = (p1..., p2...)
    sizes = [map(n -> size(block, n), ind) for block in A.blocks]
    dirs = op.(getelements(A.dirs,ind))
    qns = [getelements(qns,ind) for qns in A.qns]
    (sizes, qns, dirs, op(A.qntotal))
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
    qntotal = fuse(opA(A.qntotal),opB(B.qntotal))
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
const CovStructure{N,QN} = Tuple{Vector{NTuple{N,Int}},Vector{QN},NTuple{N,Bool},QN}
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


function SymmetricTensor(blocks::Vector, qns::Vector{QN}, dirs) where QN
    @assert all(map(qn -> iszero(fuse(qn, dirs)), qns)) "$qns \n $dirs \n $(map(qn->fuse(qn,dirs),qns) )"
    CovariantTensor(blocks, qns, dirs, zero(QN))
end


#QuantumNumber(l::Link) = l[2] ? invert(l[1]) : l[1]
Base.:(==)(a::ZQuantumNumber{N}, b::ZQuantumNumber{N}) where N = a.n == b.n
Base.hash(a::ZQuantumNumber, h::UInt64 = UInt64(0)) = Base.hash(a.n, h)
#Base.:!(qn::ParityQN) = ParityQN(!qn.parity)
#fuse(qn1::ParityQN,qn2::ParityQN) = qn1.parity + qn2.parity
invert(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
invert(::TrivialQN) = TrivialQN()
Base.zero(::Type{QN}) where QN<:Union{ZQuantumNumber,U1QuantumNumber} = QN(0)
#invert(l::ParityQN) = !l
#invert(ls::QNTuple) = QNTuple(invert.(ls.qns))
# Base.:+(lA::ZQuantumNumber{N}, lB::ZQuantumNumber{N}) where {N} = ZQuantumNumber{N}(lA.n + lB.n)
Base.:-(l::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(-l.n)
Base.:-(l::QN,l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(l.n - l2.n)
Base.:+(l::QN,l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = QN(l.n + l2.n)
Base.iszero(qn::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = iszero(qn.n)
Base.iszero(::TrivialQN) = true
fuse(l1::QN, l2::QN) where {QN<:Union{ZQuantumNumber,U1QuantumNumber}} = l1 + l2
fuse(l1::AbstractQuantumNumber, l2::TrivialQN) = l1
fuse(l1::TrivialQN, l2::AbstractQuantumNumber) = l2
fuse(l1::TrivialQN, l2::TrivialQN) = TrivialQN()
#fuse(l1::QNs, l2::QNs) where {QNs<:Tuple} = map(fuse, l1, l2)
#fuse(l1::Link{QN},l2::Link{QN}) where {QN<:AbstractQuantumNumber}= fuse(QuantumNumber(l1),QuantumNumber(l2))
#fuse(ls::NTuple{<:Any,Union{Link,AbstractQuantumNumber}}) = foldl(fuse,ls)
fuse(l::AbstractQuantumNumber, d::Bool) = d ? invert(l) : l
fuse(l1s::NTuple{N}, d1s::NTuple{N,Bool}) where {N} = fuse(map(fuse, l1s, d1s))
fuse(l1s::Tuple) = foldl(fuse, l1s)
fuse(l1s::Tuple{}) = TrivialQN()
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
TensorOperations.scalar(C::CovariantTensor) = ndims(C)==0 && C.qntotal == TrivialQN() ? sum(C.blocks)[] : throw(DimensionMismatch())
TensorOperations.memsize(a::CovariantTensor) = Base.summarysize(a)

Base.Array(A::CovariantTensor) = block_diagonal(A.blocks)

function Base.rand(::Type{NTuple{N,QN}}, qntotal::QN = zero(QN)) where {QN,N}
    qns = [rand(QN) for k in 1:N]
    qns[end] = -fuse(qns[begin:end-1])
    qns[end] = fuse(qns[end],qntotal)
    Tuple(qns)::NTuple{N,QN}
end

function Base.rand(::Type{CovariantTensor{T,N,QN}}, n::Integer = 0) where {T,N,QN}
    n = rand(1:10)
    sizes = rand(1:10,n,N)
    blocks = [rand(T,sizes[k,:]...) for k in 1:n]
    dirs = Tuple(rand(Bool,N))
   # QNstypes = fieldtypes(QNs)
    qntotal = rand(QN)
    qns = [rand(NTuple{N,QN},qntotal) for k in 1:n]
    A = CovariantTensor(blocks,qns,dirs,qntotal)::CovariantTensor{T,N,QN}
    cat_collisions!(A)
end
Base.rand(::Type{TrivialQN}) = TrivialQN()
Base.rand(::Type{ZQuantumNumber{N}}) where {N} = ZQuantumNumber{N}(rand(Mod{N}))

function rand_compatible_tensor(A::CovariantTensor{T,N,QN}) where {T,N,QN}
    n = length(A.blocks)
    sizes = size.(A.blocks)
    blocks = [rand(T,sizes[k,:]...) for k in 1:n]
    return CovariantTensor(blocks,A.qns,invert.(A.dirs),A.qntotal)#::CovariantTensor{T,N,QNs,QN}
end

function LinearAlgebra.ishermitian(A::CovariantTensor{T,N}) where {T,N}
    iseven(N) || return false
    Nhalf = Int( N/ 2)
    invert.(A.dirs[1:Nhalf]) == A.dirs[Nhalf+1:end] || return false
    all([qns[1:Nhalf] == qns[Nhalf+1:end] for qns in A.qns]) || return false
    all([b' ≈ b for b in A.blocks]) || return false
    return true
end
function isunitary(A::CovariantTensor{T,N}) where {T,N}
    iseven(N) || return false
    Nhalf = Int( N/ 2)
    #all(map(xor,A.dirs[1:Nhalf],A.dirs[Nhalf+1:end])) || return false
    all(isunitary.(A.blocks)) || return false
    #mat' * mat ≈ one(mat) && mat * mat' ≈ one(mat)
    return true
end

function Base.permutedims(A::CovariantTensor,p)
    blocks = [permutedims(block,p) for block in A.blocks]
    qns = [getelements(qns,p) for qns in A.qns]
    dirs = getelements(A.dirs,p)
    CovariantTensor(blocks,qns,dirs,A.qntotal)
end

function Base.:*(a::Number, A::CovariantTensor)
    B = similar(A)
    Threads.@threads for n in eachindex(B.blocks)
        B.blocks[n] = a * A.blocks[n]
    end
    return B
end
function LinearAlgebra.lmul!(a::Number, B::CovariantTensor)
    Threads.@threads for block in B.blocks
        lmul!(a, block)
    end
    return B
end

function LinearAlgebra.rmul!(B::CovariantTensor, a::Number)
    Threads.@threads for block in B.blocks
        lmul!(block, a)
    end
    return B
end

function LinearAlgebra.mul!(w::CovariantTensor, a::Number, v::CovariantTensor)
    @assert w.dirs == v.dirs && w.qns == v.qns
    Threads.@threads for (wb,vb) in zip(w.blocks,v.blocks)
        mul!(wb, a, vb)
    end
    return w
end

function LinearAlgebra.mul!(w::CovariantTensor, v::CovariantTensor, a::Number)
    @assert w.dirs == v.dirs && w.qns == v.qns
    Threads.@threads for (wb,vb) in zip(w.blocks,v.blocks)
        mul!(wb, vb, a)
    end
    return w
end

function LinearAlgebra.axpy!(a::Number, v::CovariantTensor, w::CovariantTensor)
    @assert w.dirs == v.dirs && w.qns == v.qns
    Threads.@threads for (wb,vb) in zip(w.blocks,v.blocks)
        axpy!(a, vb,wb)
    end
    return w
end
function LinearAlgebra.axpby!(a::Number, v::CovariantTensor, b, w::CovariantTensor)
    @assert w.dirs == v.dirs && w.qns == v.qns
    Threads.@threads for (wb,vb) in zip(w.blocks,v.blocks)
        axpby!(a, vb,b, wb)
    end
    return w
end

function LinearAlgebra.dot(v::CovariantTensor{Tv,N}, w::CovariantTensor{Tw,N}) where {N,Tv,Tw}
    cind::IndexTuple{N} = Tuple(1:N)
    @assert v.dirs == w.dirs
    @assert length(v.qns) == length(w.qns)
    pairs = find_matches(v,cind,w,cind)
    T = promote_type(Tv,Tw)
    T0 = fill(zero(T))
    C = [T0 for k in 1:N]
    for (nC,(nv,nw)) in enumerate(pairs)
        TensorOperations.contract!(one(T),v.blocks[nv],:C,w.blocks[nw],:N,zero(T),C[nC],(),cind,(),cind,(),())
    end
    return scalar(sum(C))
end
LinearAlgebra.norm(v::CovariantTensor) = norm(norm.(v.blocks))

# Base.conj(A::CovariantTensor)

