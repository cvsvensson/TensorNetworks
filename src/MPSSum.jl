
struct MPSSum{MPS<:AbstractMPS,Site<:AbstractSite,Num} <: AbstractMPS{Site}
    states::Vector{MPS}
    scalings::Vector{Num}
    function MPSSum(mpss::Vector{T}) where {T<:AbstractMPS}
        new{T,SiteSum{eltype(mpss[1]),numtype(mpss...)},numtype(mpss...)}([mpss...], fill(1, length(mpss)))
    end
    function MPSSum(mpss::Vector{T}, scalings::Vector{Num}) where {T<:AbstractMPS,Num}
        new{T,SiteSum{eltype(mpss[1]),numtype(mpss...)},numtype(mpss...)}(mpss, scalings)
    end
end
Base.show(io::IO, mps::MPSSum) =
    (print(io, "MPS: ", typeof(mps), "\nSites: ", eltype(mps), "\nLength: ", length(mps), "\nSum of ", length(mps.states), " mps's\nWith scalings "); show(io, mps.scalings))
Base.show(io::IO, m::MIME"text/plain", mps::MPSSum) = show(io, mps)
Base.size(mps::MPSSum) = (length(mps),)
Base.length(mps::MPSSum) = length(mps.states[1])
Base.copy(mps::MPSSum) = MPSSum(copy(mps.states), copy(mps.scalings))

struct SiteSum{S<:AbstractSite,T} <: AbstractCenterSite{T}
    sites::Vector{S}
    function SiteSum(sites::Vector{S}) where {S<:AbstractCenterSite}
        new{S,eltype(sites[1])}(sites)
    end
end
SiteSum(site::AbstractCenterSite) = SiteSum([site])
Base.show(io::IO, mps::SiteSum) =
    print(io, "SiteSum: ", typeof(mps), "\nSites: ", eltype(mps), "\nLength: ", length(mps.sites))
Base.show(io::IO, m::MIME"text/plain", mps::SiteSum) = show(io, mps)

Base.size(sites::SiteSum) = (sum(size.(sites.sites, 1)), size(sites.sites[1], 2), sum(size.(sites.sites, 3)))
Base.length(sites::SiteSum) = length(sites.sites)
Base.size(sites::SiteSum, i::Integer) = (sum(size.(sites.sites, 1)), size(sites.sites[1], 2), sum(size.(sites.sites, 3)))[i]

Base.conj(site::SiteSum) = SiteSum(conj.(site.sites))

Base.:+(mps1::AbstractMPS, mps2::AbstractMPS) = MPSSum([mps1, mps2])
Base.:+(mps::AbstractMPS, sum::MPSSum) = 1 * mps + sum
Base.:+(sum::MPSSum, mps::AbstractMPS) = sum + 1 * mps
Base.:+(s1::MPSSum, s2::MPSSum) = MPSSum(vcat(s1.states, s2.states), vcat(s1.scalings, s2.scalings))

Base.:+(s1::AbstractSite, s2::AbstractSite) = SiteSum([s1, s2])

Base.:*(x::Number, mps::AbstractMPS) = MPSSum([mps], [x])
Base.:*(mps::AbstractMPS, x::Number) = MPSSum([mps], [x])
Base.:*(x::Number, mps::MPSSum) = MPSSum(mps.states, x * mps.scalings)
Base.:*(mps::MPSSum, x::Number) = MPSSum(mps.states, x * mps.scalings)
Base.:/(mps::MPSSum, x::Number) = MPSSum(mps.states, inv(x) * mps.scalings)
Base.:-(mps::AbstractMPS) = (-1) * mps
Base.:-(mps::AbstractMPS, mps2::AbstractMPS) = mps + (-1) * mps2

Base.IndexStyle(::Type{<:MPSSum}) = IndexLinear()
Base.getindex(sum::MPSSum, i::Integer) = SiteSum([state[i] for state in sum.states])

Base.getindex(sum::SiteSum, i::Integer) = sum.sites[i]
Base.IndexStyle(::Type{<:SiteSum}) = IndexLinear()
reverse_direction(sitesum::SiteSum) = SiteSum(reverse_direction.(sitesum.sites))
ispurification(sitesum::SiteSum) = ispurification(sitesum[1])


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
    Ts = [_transfer_right_gate(getindex.(Γ1, n1), gate, getindex.(Γ2, n2)) for n1 in 1:N1, n2 in 1:N2]
    return blockdiagonal(Ts) #LinearMaps.blockdiag(Ts...) #_apply_transfer_matrices(Ts)
end
_transfer_right_gate(Γ1::Vector{<:AbstractSite}, gate::GenericSquareGate, Γ2::Vector{<:SiteSum}) = _transfer_right_gate(SiteSum.(Γ1), gate, Γ2)
_transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate, Γ2::Vector{<:AbstractSite}) = _transfer_right_gate(Γ1, gate, SiteSum.(Γ2))
_transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate) = _transfer_right_gate(Γ1, gate, Γ1)


_transfer_left_mpo(Γ1::GenericSite, op, Γ2::SiteSum) = _transfer_left_mpo(SiteSum(Γ1), op, Γ2)
_transfer_left_mpo(Γ1::SiteSum, op, Γ2::GenericSite) = _transfer_left_mpo(Γ1, op, SiteSum(Γ2))
_transfer_left_mpo(Γ1::GenericSite, Γ2::SiteSum) = _transfer_left_mpo(SiteSum(Γ1), Γ2)
_transfer_left_mpo(Γ1::SiteSum, Γ2::GenericSite) = _transfer_left_mpo(Γ1, SiteSum(Γ2))
_transfer_left_mpo(Γ1::SiteSum, op::MPOsite) = _transfer_left_mpo(Γ1, op, Γ1)
function _transfer_left_mpo(Γ1::SiteSum, op, Γ2::SiteSum)
    N1 = length(Γ1)
    N2 = length(Γ2)
    Ts = [_transfer_left_mpo(Γ1[n1], op, Γ2[n2]) for n1 in 1:N1, n2 in 1:N2]
    return blockdiagonal(Ts)
    #return _apply_transfer_matrices(Ts)
end
function _transfer_left_mpo(Γ1::SiteSum, Γ2::SiteSum)
    N1 = length(Γ1)
    N2 = length(Γ2)
    Ts = [_transfer_left_mpo(Γ1[n1], Γ2[n2]) for n1 in 1:N1, n2 in 1:N2]
    return blockdiagonal(Ts)
    #return _apply_transfer_matrices(Ts)
end
function _transfer_left_mpo(Γ1::SiteSum)
    N1 = length(Γ1)
    Ts = [_transfer_left_mpo(Γ1[n1], Γ1[n2]) for n1 in 1:N1, n2 in 1:N1]
    return blockdiagonal(Ts)
    #return LinearMaps.blockdiag(Ts...)
    #return _apply_transfer_matrices(Ts)
end
blockdiagonal(Ts::Array{<:LinearMap{T}}) where T = LinearMaps.BlockDiagonalMap{T}(vec(Ts))

# function _transfer_left_mpo(Γ::AbstractSite)
#     Ts = [_transfer_left_mpo(block) for block in blocks(Γ)]
# end



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
    return LinearMap{eltype(Ts[1, 1])}(f, f_adjoint, DL, DR)
end

boundaryconditions(::Type{<:MPSSum{M,S}}) where {M,S} = boundaryconditions(M)
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
transfer_matrix_bond(mps::AbstractVector{<:SiteSum{<:GenericSite,<:Any}}, site::Integer, dir::Symbol) = I


function dense(mpss::MPSSum{LCROpenMPS{T},<:Any}) where {T}
    sites = dense.(mpss)
    sites[1] = (mpss.scalings) * sites[1]
    sites[end] = sites[end] * (ones(T, length(mpss.scalings)))
    return LCROpenMPS{T}(to_left_right_orthogonal(sites), truncation = mpss.states[1].truncation, error = sum(getproperty.(mpss.states, :error)))
end

function dense(sitesum::SiteSum{<:GenericSite,T}) where {T}
    sizes = size.(sitesum.sites)
    d = sizes[1][2] #Maybe check that all sites have the same physical dim?
    DL = sum([s[1] for s in sizes])
    DR = sum([s[3] for s in sizes])
    newsite = zeros(T, DL, d, DR)
    lastL = 0
    lastR = 0
    for (site, size) in zip(sitesum.sites, sizes)
        nextL = lastL + size[1]
        nextR = lastR + size[3]
        newsite[lastL+1:nextL, :, lastR+1:nextR] = data(site)
        lastL = nextL
        lastR = nextR
    end
    return GenericSite(newsite, ispurification(sitesum))
end

function dense(sitesum::SiteSum{<:OrthogonalLinkSite,T}) where {T}
    Γ = dense(SiteSum(getproperty.(sitesum.sites, :Γ)))
    Λ1s = getproperty.(sitesum.sites, :Λ1)
    Λ2s = getproperty.(sitesum.sites, :Λ2)
    Λ1 = LinkSite(reduce(vcat, vec.(Λ1s)))
    Λ2 = LinkSite(reduce(vcat, vec.(Λ2s)))
    return OrthogonalLinkSite(Λ1, Γ, Λ2)
end

function dense(mpss::MPSSum{OpenMPS{T},<:Any}) where {T}
    lcrsum = MPSSum(LCROpenMPS.(mpss.states), mpss.scalings)
    denselcr = dense(lcrsum)
    OpenMPS(denselcr)
end

Base.vec(site::LinkSite) = vec(diag(data(site)))

function transfer_matrix_bond(mps::AbstractVector{<:SiteSum{<:OrthogonalLinkSite,<:Any}}, site::Integer, dir::Symbol)
    Λ1s = getproperty.(mps[site].sites, :Λ1)
    return LinkSite(reduce(vcat, vec.(Λ1s)))
end

iscanonical(sites::SiteSum) = all(iscanonical.(sites.sites))