Base.similar(site::SiteSum) = SiteSum(similar.(site.sites))
Base.copy(site::SiteSum) = SiteSum(copy.(site.sites))
Base.:*(x::Number, site::SiteSum) = SiteSum(x .* sites(site))
Base.:*(site::SiteSum, x::Number) = x * site
Base.:/(site::SiteSum, x::Number) = inv(x) * site
#Base.setindex!(site::GenericSite, v, I::Vararg{Integer,3}) = (data(site)[I...] = v)
function LinearAlgebra.mul!(w::SiteSum, v::SiteSum, x::Number)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([mul!(sw, sv, x) for (sv, sw) in zip(sites(v), sites(w))])
end
LinearAlgebra.rmul!(v::SiteSum, x::Number) = SiteSum([rmul!(sv, x) for sv in sites(v)])
LinearAlgebra.norm(v::SiteSum) = norm(norm.(sites(v)))
LinearAlgebra.dot(v::SiteSum, w::SiteSum) = sum([dot(sv, sw) for (sv, sw) in zip(sites(v), sites(w))])

function LinearAlgebra.axpy!(x::Number, v::SiteSum, w::SiteSum)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([axpy!(x, sv, sw) for (sv, sw) in zip(sites(v), sites(w))])
end
function LinearAlgebra.axpby!(x::Number, v::SiteSum, β::Number, w::SiteSum)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([axpby!(x, sv, β, sw) for (sv, sw) in zip(sites(v), sites(w))])
end

Base.:*(op::MPOsite, sites::SiteSum) = SiteSum([op * site for site in sites.sites])
multiply(op::MPOsite, site::SiteSum) = SiteSum(Tuple([multiply(op, s) for s in sites(site)]))

#TODO: change vector argument to tuple arguments.
#MPSSum(sites::NTuple{<:Any,<:AbstractMPS}) = MPSSum([sites...])
#MPSSum(sites::NTuple{<:Any,<:AbstractMPS},s) = MPSSum([sites...],s)
Base.show(io::IO, mps::MPSSum) =
    (print(io, "MPS: ", typeof(mps), "\nSites: ", eltype(mps), "\nLength: ", length(mps), "\nSum of ", length(mps.states), " mps's\nWith scalings "); show(io, mps.scalings))
Base.show(io::IO, m::MIME"text/plain", mps::MPSSum) = show(io, mps)
Base.size(mps::MPSSum) = (length(mps),)
Base.length(mps::MPSSum) = length(mps.states[1])
Base.copy(mps::MPSSum) = MPSSum(copy(mps.states), copy(mps.scalings))


SiteSum(site::AbstractCenterSite) = SiteSum((site,))
#SiteSum(sites::Tuple) = SiteSum([sites...])
Base.show(io::IO, mps::SiteSum) =
    print(io, "SiteSum: ", typeof(mps), "\nSites: ", eltype(mps), "\nLength: ", length(mps.sites))
Base.show(io::IO, m::MIME"text/plain", mps::SiteSum) = show(io, mps)

Base.size(sites::SiteSum) = (sum(size.(sites.sites, 1)), size(sites.sites[1], 2), sum(size.(sites.sites, 3)))
Base.length(sites::SiteSum) = length(sites.sites)
Base.size(sites::SiteSum, i::Integer) = (sum(size.(sites.sites, 1)), size(sites.sites[1], 2), sum(size.(sites.sites, 3)))[i]

Base.conj(site::SiteSum) = SiteSum(conj.(site.sites))

Base.:+(mps1::AbstractMPS, mps2::AbstractMPS) = MPSSum((mps1, mps2))
Base.:+(mps::AbstractMPS, sum::MPSSum) = 1 * mps + sum
Base.:+(sum::MPSSum, mps::AbstractMPS) = sum + 1 * mps
Base.:+(s1::MPSSum, s2::MPSSum) = MPSSum(tuple(s1.states..., s2.states...), vcat(s1.scalings, s2.scalings))

Base.:+(s1::AbstractSite, s2::AbstractSite) = SiteSum((s1, s2))

Base.:*(x::Number, mps::AbstractMPS) = MPSSum((mps,), [x])
Base.:*(mps::AbstractMPS, x::Number) = x * mps
Base.:*(x::Number, mps::MPSSum) = MPSSum(mps.states, x * mps.scalings)
Base.:*(mps::MPSSum, x::Number) = x * mps
Base.:/(mps::MPSSum, x::Number) = inv(x) * mps
Base.:-(mps::AbstractMPS) = (-1) * mps
Base.:-(mps::AbstractMPS, mps2::AbstractMPS) = mps + (-1) * mps2

Base.IndexStyle(::Type{<:MPSSum}) = IndexLinear()
SiteSum(sites...) = SiteSum(sites)
Base.getindex(sum::MPSSum, i::Integer) = SiteSum((state[i] for state in sum.states)...)
Base.getindex(sum::MPSSum, I...) = (mapfoldr(mps -> mps[I...], .+, sum.states))

#Base.getindex(sum::SiteSum, i::Integer) = sum.sites[i]
#Base.IndexStyle(::Type{<:SiteSum}) = IndexLinear()
reverse_direction(sitesum::SiteSum) = SiteSum(reverse_direction.(sitesum.sites))
ispurification(sitesum::SiteSum) = ispurification(sitesum.sites[1])


function Base.setindex!(mps::MPSSum, v::SiteSum, i::Integer)
    @assert length(v) == length(mps[i]) "Error: incompatible number of sites in setindex!"
    for n in 1:length(v)
        mps[i].sites[n] = v[n]
    end
    return v
end


_split_lazy_v(s, ss::Tuple) = (s, ss...)
_split_lazy_v(ss::Tuple, s) = (ss..., s)
_split_lazy_v(s::Vector{<:LazySiteProduct}, ss::Tuple) = ([[mp] for mp in s[1].sites]..., ss...)
_split_lazy_v(ss::Tuple, s::Vector{<:LazySiteProduct}) = (ss..., [[mp] for mp in reverse(s[1].sites)]...)
# function itr_split_lazy(s::Tuple)
#     [sites(s) for site in s]
# end

function _transfer_left_mpo(csites::Tuple, s::Tuple)
    csites2 = foldl(_split_lazy_v, sites.(csites), init=())
    sites2 = foldr(_split_lazy_v, sites.(s), init=())
    #K = promote_type(eltype.(s)...,eltype.(csites)...)
    #N = stackheight(csites) + stackheight(s)
    __transfer_left_mpo(csites2, sites2)#::TransferMatrix{K,N,NTuple{2,Array{NTuple{N,Int},N}}}
end
sites(s::Tuple) = mapfoldr(sites, _split_lazy_v, s, init=())

# _split_lazy_v(s::Vector{<:LazySiteProduct}, ss::Tuple) = ([[mp] for mp in s[1].sites]..., ss...)
# _split_lazy_v(ss::Tuple, s::Vector{<:LazySiteProduct}) = (ss..., [[mp] for mp in reverse(s[1].sites)]...,)

function __transfer_left_mpo(csites::Tuple, s::Tuple)
    K = mapreduce(s -> eltype(s[1]), promote_type, (csites..., s...))
    #K = promote_type(eltype.(s)...,eltype.(csites)...)
    N = length(csites) + length(s)
    itr = Base.product(Base.product((csites)...), Base.product((s)...))
    #println(collect(itr)[1])
    Ts::Array{TransferMatrix{K,N,NTuple{2,NTuple{N,Int}}},N} = [_transfer_left_mpo_dense(cs, ss) for (cs, ss) in itr]
    _apply_transfer_matrices(Ts)::TransferMatrix{K,N,NTuple{2,Array{NTuple{N,Int},N}}}
end

function transfer_matrix_bond(csites::Tuple, s::Tuple)
    csites2 = foldl(_split_lazy_v, sites.(csites), init=())
    sites2 = foldr(_split_lazy_v, sites.(s), init=())
    K = promote_type(eltype.(s)..., eltype.(csites)...)
    #N = length(csites2) + length(sites2)
    N = stackheight(csites) + stackheight(s)
    _transfer_matrix_bond(csites2, sites2)::TransferMatrix{K,N,NTuple{2,Array{NTuple{N,Int},N}}}
end
function _transfer_matrix_bond(csites::Tuple, s::Tuple)
    itr = Base.product(Base.product((csites)...), Base.product((s)...))
    Ts = [transfer_matrix_bond_dense(cs, ss) for (cs, ss) in itr]
    _apply_transfer_matrices(Ts) #_apply_transfer_matrices([transfer_matrix_bond(ss...) for ss in Base.product(sites.(ss)...)])
end
function _split_vector(v, s1::NTuple{N1,Int}, s2::NTuple{N2,Int}) where {N1,N2}
    tens = Matrix{Vector{eltype(v)}}(undef, (N1, N2))
    last = 0
    for n2 in 1:N2
        for n1 in 1:N1
            next = last + s1[n1] * s2[n2]
            tens[n1, n2] = v[last+1:next]
            last = next
        end
    end
    return tens
end
function _split_vector(v, s::Array{NTuple{N,Int},K}) where {N,K}
    ranges = size(s)
    tens = Array{Array{eltype(v),N},K}(undef, ranges)
    last = 0
    for ns in Base.product((1:r for r in ranges)...)
        next = last + prod(s[ns...])
        tens[ns...] = reshape(v[last+1:next], s[ns...])
        last = next
    end
    return tens
end
_split_vector(v, s::Dims) = reshape(v, s...)

function _join_tensor(tens)
    reduce(vcat, tens)
end


#boundaryconditions(::Type{<:MPSSum{M,S}}) where {M,S} = boundaryconditions(M)
function boundaryconditions(mps::MPSSum)
    bcs = boundaryconditions.(mps.states)
    @assert length(union(bcs)) == 1
    return bcs[1]
end
#,LazyProduct{<:Any,<:Any,<:MPOSum}
function boundary(::OpenBoundary, mps::Union{MPSSum}, side::Symbol)
    if side == :right
        return BlockBoundaryVector([boundary(OpenBoundary(), mps.states[k], :right) for k in 1:length(mps.states)])
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        # v=[mps.scalings[k] .* boundary(OpenBoundary(), mps.states[k], :left) for k in 1:length(mps.states)]
        # println.(typeof.(v))
        # println("ASD")
        # bvs = BlockBoundaryVector.(v)
        # println.(bvs)
        # println.(eltype.(BlockBoundaryVector.(v)))
        # println(BlockBoundaryVector(v))
        return BlockBoundaryVector([mps.scalings[k] * boundary(OpenBoundary(), mps.states[k], :left) for k in 1:length(mps.states)])
    end
end

truncation(mps::MPSSum) = truncation(mps.states[1])

function dense(mpss::MPSSum{<:NTuple{<:Any,<:LCROpenMPS{T}},<:Any}) where {T}
    sites = dense.(mpss)
    sites[1] = (mpss.scalings) * sites[1]
    sites[end] = sites[end] * (ones(T, length(mpss.scalings)))
    return LCROpenMPS{T}(to_left_right_orthogonal(sites), truncation=truncation(mpss.states[1]), error=sum(error.(mpss.states)))
end

function LCROpenMPS(mpss::MPSSum{<:Any,<:AbstractSite{T},<:Any}) where {T}
    sites = dense.(mpss)
    sites[1] = (mpss.scalings) * sites[1]
    sites[end] = sites[end] * (ones(T, length(mpss.scalings)))
    return LCROpenMPS{T}(to_left_right_orthogonal(sites), truncation=truncation(mpss.states[1]), error=sum(error.(mpss.states)))
end

# function dense(sitesum::SiteSum{<:NTuple{<:Any,GenericSite},T}) where {T}
#     sizes = size.(sitesum.sites)
#     d = sizes[1][2] #Maybe check that all sites have the same physical dim?
#     DL = sum([s[1] for s in sizes])
#     DR = sum([s[3] for s in sizes])
#     newsite = zeros(T, DL, d, DR)
#     lastL = 0
#     lastR = 0
#     for (site, size) in zip(sitesum.sites, sizes)
#         nextL = lastL + size[1]
#         nextR = lastR + size[3]
#         newsite[lastL+1:nextL, :, lastR+1:nextR] = data(site)
#         lastL = nextL
#         lastR = nextR
#     end
#     return GenericSite(newsite, ispurification(sitesum))
# end

function dense(sitesum::SiteSum{Tup,T}) where {Tup,T}
    sites = dense.(sitesum.sites)
    sizes = size.(sites)
    d = sizes[1][2] #Maybe check that all sites have the same physical dim?
    DL = sum([s[1] for s in sizes])
    DR = sum([s[3] for s in sizes])
    newsite = zeros(T, DL, d, DR)
    lastL = 0
    lastR = 0
    for (site, size) in zip(sites, sizes)
        nextL = lastL + size[1]
        nextR = lastR + size[3]
        newsite[lastL+1:nextL, :, lastR+1:nextR] = data(site)
        lastL = nextL
        lastR = nextR
    end
    return GenericSite(newsite, ispurification(sitesum))
end
Base.convert(::Type{<:GenericSite}, site::SiteSum) = dense(site)
GenericSite(s::SiteSum) = dense(s)
dense(s::GenericSite) = s

function dense(sitesum::SiteSum{<:NTuple{<:Any,OrthogonalLinkSite},T}) where {T}
    Γ = dense(SiteSum(getproperty.(sitesum.sites, :Γ)))
    Λ1s = getproperty.(sitesum.sites, :Λ1)
    Λ2s = getproperty.(sitesum.sites, :Λ2)
    Λ1 = LinkSite(reduce(vcat, vec.(Λ1s)))
    Λ2 = LinkSite(reduce(vcat, vec.(Λ2s)))
    return OrthogonalLinkSite(Λ1, Γ, Λ2)
end

function dense(mpss::MPSSum{<:NTuple{<:Any,<:OpenMPS},<:Any})
    lcrsum = MPSSum(LCROpenMPS.(mpss.states), mpss.scalings)
    denselcr = dense(lcrsum)
    OpenMPS(denselcr)
end

Base.vec(site::LinkSite) = vec(diag(data(site)))

# function transfer_matrix_bond(sites::SiteSum)
#     Λ1s = transfer_matrix_bond_dense.(sites.sites)
#     #println(Λ1s)
#     #println( reduce(vcat,vec.(Λ1s)))
#     return LinkSite(vec(reduce(vcat, Array.(Λ1s))))
# end

# transfer_matrix_bond(mps::AbstractVector{<:SiteSum{<:NTuple{N,<:GenericSite},<:Any}}, site::Integer) where N = IdentityTransferMatrix
#transfer_matrix_bond(mps::MPSSum{<:NTuple{<:Any,<:LCROpenMPS},<:Any}, site::Integer, dir::Symbol) = I
# transfer_matrix_bond(mps::SiteSum{<:NTuple{<:Any,<:GenericSite},T}) where T = I #IdentityTransferMatrix(T)
# transfer_matrix_bond_dense(mps::SiteSum) = LinkSite(vec(reduce(vcat, Array.(transfer_matrix_bond_dense.(mps.sites,dir)))))

iscanonical(sites::SiteSum) = all(iscanonical.(sites.sites))