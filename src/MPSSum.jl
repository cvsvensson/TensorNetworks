struct SiteSum{S<:Tuple,T} <: AbstractCenterSite{T}
    sites::S
    function SiteSum(sites::Tuple)
        #println(typeof(Tuple((sites))), promote_rule(eltype.(sites)...))
        new{typeof(sites), promote_type(eltype.(sites)...)}(copy.(sites))
    end
end
Base.similar(site::SiteSum) = SiteSum(similar.(site.sites))
Base.copy(site::SiteSum) = SiteSum(copy.(site.sites))
Base.:*(x::Number, site::SiteSum) = SiteSum(x .* sites(site))
#Base.setindex!(site::GenericSite, v, I::Vararg{Integer,3}) = (data(site)[I...] = v)
function LinearAlgebra.mul!(w::SiteSum, v::SiteSum, x::Number)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([mul!(sw, sv, x) for (sv,sw) in zip(sites(v), sites(w))])
end
LinearAlgebra.rmul!(v::SiteSum, x::Number) = SiteSum([rmul!(sv, x) for sv in sites(v)])
LinearAlgebra.norm(v::SiteSum) = norm(norm.(sites(v)))
LinearAlgebra.dot(v::SiteSum,w::SiteSum) = sum([dot(sv,sw) for (sv,sw) in zip(sites(v),sites(w))])

function LinearAlgebra.axpy!(x::Number, v::SiteSum, w::SiteSum)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([axpy!(x, sv, sw) for (sv,sw) in zip(sites(v),sites(w))])
end
function LinearAlgebra.axpby!(x::Number, v::SiteSum, β::Number, w::SiteSum)
    @assert length(sites(w)) == length(sites(v)) "Error: Storage is differently sized from input"
    SiteSum([axpby!(x, sv, β, sw) for (sv,sw) in zip(sites(v),sites(w))])
end

Base.:*(op::MPOsite, sites::SiteSum) = SiteSum([op * site for site in sites.sites])


struct MPSSum{MPSs<:Tuple,Site<:AbstractSite,Num} <: AbstractMPS{Site}
    states::MPSs
    scalings::Vector{Num}
    function MPSSum(mpss::Tuple, scalings::Vector{Num}) where {Num}
        new{typeof(mpss),SiteSum{Tuple{eltype.(mpss)...},numtype(mpss...)},numtype(mpss...)}(Tuple(mpss), scalings)
    end
end
function MPSSum(mpss::Tuple)
    MPSSum(mpss, fill(one(numtype(mpss...)), length(mpss)))
end
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
Base.:*(mps::AbstractMPS, x::Number) = MPSSum((mps,), [x])
Base.:*(x::Number, mps::MPSSum) = MPSSum(mps.states, x * mps.scalings)
Base.:*(mps::MPSSum, x::Number) = MPSSum(mps.states, x * mps.scalings)
Base.:/(mps::MPSSum, x::Number) = MPSSum(mps.states, inv(x) * mps.scalings)
Base.:-(mps::AbstractMPS) = (-1) * mps
Base.:-(mps::AbstractMPS, mps2::AbstractMPS) = mps + (-1) * mps2

Base.IndexStyle(::Type{<:MPSSum}) = IndexLinear()
Base.getindex(sum::MPSSum, i::Integer) = SiteSum(Tuple(state[i] for state in sum.states))

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

function _transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate, Γ2::Vector{<:SiteSum})
    #FIXME Might screw up type stability, so might need a TransferMatrix struct?
    N1 = length(Γ1[1])
    N2 = length(Γ2[1])
    Ts = [_transfer_right_gate(map(s->sites(s)[n1],Γ1), gate, map(s->sites(s)[n2],Γ2)) for n1 in 1:N1, n2 in 1:N2]
    return _apply_transfer_matrices(Ts)
end
_transfer_right_gate(Γ1::Vector{<:AbstractSite}, gate::GenericSquareGate, Γ2::Vector{<:SiteSum}) = _transfer_right_gate(SiteSum.(Γ1), gate, Γ2)
_transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate, Γ2::Vector{<:AbstractSite}) = _transfer_right_gate(Γ1, gate, SiteSum.(Γ2))
_transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate) = _transfer_right_gate(Γ1, gate, Γ1)


_transfer_left_mpo(Γ1::GenericSite, op, Γ2::SiteSum) = _transfer_left_mpo(SiteSum(Γ1), op, Γ2)
_transfer_left_mpo(Γ1::SiteSum, op, Γ2::GenericSite) = _transfer_left_mpo(Γ1, op, SiteSum(Γ2))
_transfer_left_mpo(Γ1::GenericSite, Γ2::SiteSum) = _transfer_left_mpo(SiteSum(Γ1), Γ2)
_transfer_left_mpo(Γ1::SiteSum, Γ2::GenericSite) = _transfer_left_mpo(Γ1, SiteSum(Γ2))
_transfer_left_mpo(Γ1::SiteSum, op::MPOsite) = _transfer_left_mpo(Γ1, op, Γ1)
_transfer_left_mpo(Γ1::SiteSum) = _transfer_left_mpo(Γ1, Γ1)

function _transfer_left_mpo(Γ1, op, Γ2)
    Ts = [_transfer_left_mpo(Γ1site, opsite, Γ2site) for Γ1site in sites(Γ1), opsite in sites(op), Γ2site in sites(Γ2)]
    return _apply_transfer_matrices(Ts)
end
function _transfer_left_mpo(Γ1, Γ2)
    Ts = [_transfer_left_mpo(Γ1site, Γ2site) for Γ1site in sites(Γ1), Γ2site in sites(Γ2)]
    return _apply_transfer_matrices(Ts)
end

# function _transfer_left_mpo(Γ1::SiteSum, op, Γ2::SiteSum)
#     N1 = length(Γ1)
#     N2 = length(Γ2)
#     Ts = [_transfer_left_mpo(Γ1[n1], op, Γ2[n2]) for n1 in 1:N1, n2 in 1:N2]
#     return _apply_transfer_matrices(Ts)
# end
# function _transfer_left_mpo(Γ1::SiteSum, Γ2::SiteSum)
#     N1 = length(Γ1)
#     N2 = length(Γ2)
#     Ts = [_transfer_left_mpo(Γ1[n1], Γ2[n2]) for n1 in 1:N1, n2 in 1:N2]
#     return _apply_transfer_matrices(Ts)
# end
# function _transfer_left_mpo(Γ1::SiteSum)
#     N1 = length(Γ1)
#     Ts = [_transfer_left_mpo(Γ1[n1], Γ1[n2]) for n1 in 1:N1, n2 in 1:N1]
#     return _apply_transfer_matrices(Ts)
# end



# function _transfer_left_mpo(Γ::AbstractSite)
#     Ts = [_transfer_left_mpo(block) for block in blocks(Γ)]
# end


#TODO: replace with Blockdiagonals.jl or BlockArrays.jl.
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
function _split_vector(v, s::Matrix{Int})
    N1, N2 = size(s)
    tens = Matrix{Vector{eltype(v)}}(undef, size(s))
    last = 0
    for n2 in 1:N2
        for n1 in 1:N1
            next = last + s[n1, n2]
            tens[n1, n2] = v[last+1:next]
            last = next
        end
    end
    return tens
end
function _split_vector(v, s::Array{Int,3})
    N1, N2, N3 = size(s)
    tens = Array{Vector{eltype(v)},3}(undef, size(s))
    last = 0
    for n3 in 1:N3
        for n2 in 1:N2
            for n1 in 1:N1
                next = last + s[n1, n2, n3]
                tens[n1, n2, n3] = v[last+1:next]
                last = next
            end
        end
    end
    return tens
end
function _split_vector(v, s::Array{NTuple{N,Int},K}) where {N,K}
    ranges = size(s)
    tens = Array{Array{eltype(v),N},K}(undef, ranges)
    last = 0
    println("V:", length(v))
    println("sizes:", s)
    for ns in Base.product((1:r for r in ranges)...)
        next = last + prod(s[ns...])
        println(ns)
        println("S: ", s[ns...])
        tens[ns...] = reshape(v[last+1:next], s[ns...])
        last = next
    end
    return tens
end
function _join_tensor(tens)
    reduce(vcat, tens)
end

function _apply_transfer_matrices(Ts)
    # N1, N2 = size(Ts)
    # # sizes = [size(Γ1[n1],3)*size(Γ1[n2],3) for n1 in 1:N1, n2 in 1:N1]
    sizes = [size(T, 2) for T in Ts]
    DL = sum(size.(Ts, 1))
    DR = sum(size.(Ts, 2))
    function f(v)
        tens = _split_vector(v, sizes)
        _join_tensor([T * t for (T, t) in zip(Ts, tens)])
    end
    function f_adjoint(v)
        tens = _split_vector(v, sizes)
        _join_tensor([T' * t for (T, t) in zip(Ts, tens)])
    end
    return LinearMap{eltype(Ts[1])}(f, f_adjoint, DL, DR)
end

#boundaryconditions(::Type{<:MPSSum{M,S}}) where {M,S} = boundaryconditions(M)
function boundaryconditions(mps::MPSSum)
    bcs = boundaryconditions.(mps.states)
    @assert length(union(bcs)) == 1
    return bcs[1]
end

function boundary(::OpenBoundary, mps::MPSSum, side::Symbol)
    if side == :right
        return fill(one(eltype(mps.scalings)), length(mps.scalings))
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return mps.scalings
    end
end

truncation(mps::MPSSum) = truncation(mps.states[1])

function dense(mpss::MPSSum{<:NTuple{<:Any,<:LCROpenMPS{T}},<:Any}) where {T}
    sites = dense.(mpss)
    sites[1] = (mpss.scalings) * sites[1]
    sites[end] = sites[end] * (ones(T, length(mpss.scalings)))
    return LCROpenMPS{T}(to_left_right_orthogonal(sites), truncation = mpss.states[1].truncation, error = sum(error.(mpss.states)))
end

function LCROpenMPS(mpss::MPSSum{<:Any,<:AbstractSite{T},<:Any}) where {T}
    sites = dense.(mpss)
    sites[1] = (mpss.scalings) * sites[1]
    sites[end] = sites[end] * (ones(T, length(mpss.scalings)))
    return LCROpenMPS{T}(to_left_right_orthogonal(sites), truncation = mpss.states[1].truncation, error = sum(error.(mpss.states)))
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
    sizes = size.(sitesum.sites)
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


transfer_matrix_bond(mps::MPSSum, site::Integer, dir::Symbol) = transfer_matrix_bond(mps[site],dir)
function transfer_matrix_bond(sites::SiteSum, dir::Symbol)
    Λ1s = transfer_matrix_bond_dense.(sites.sites, dir)
    #println(Λ1s)
    #println( reduce(vcat,vec.(Λ1s)))
    return LinkSite(vec(reduce(vcat, Array.(Λ1s))))
end

transfer_matrix_bond(mps::AbstractVector{<:SiteSum{<:NTuple{N,<:GenericSite},<:Any}}, site::Integer, dir::Symbol) where N = I
#transfer_matrix_bond(mps::MPSSum{<:NTuple{<:Any,<:LCROpenMPS},<:Any}, site::Integer, dir::Symbol) = I
transfer_matrix_bond(mps::SiteSum{<:NTuple{<:Any,<:GenericSite},<:Any}, dir::Symbol) = I
transfer_matrix_bond_dense(mps::SiteSum, dir::Symbol) = LinkSite(vec(reduce(vcat, Array.(transfer_matrix_bond_dense.(mps.sites,dir)))))


iscanonical(sites::SiteSum) = all(iscanonical.(sites.sites))