mutable struct TruncationArgs
    Dmax::Int
    tol::Float64
    normalize::Bool
end
Base.copy(ta::TruncationArgs) = TruncationArgs([copy(getfield(ta, k)) for k = 1:length(fieldnames(TruncationArgs))]...)

abstract type AbstractGate{T,N} <: AbstractArray{T,N} end
abstract type AbstractSquareGate{T,N} <: AbstractGate{T,N} end

struct ScaledIdentityGate{T,N} <: AbstractSquareGate{T,N}
    data::T
    ishermitian::Bool
    isunitary::Bool
    function ScaledIdentityGate(scaling::T, n::Integer) where {T}
        new{T,2 * n}(scaling, isreal(scaling), scaling' * scaling ≈ 1)
    end
end
IdentityGate(n) = ScaledIdentityGate(true, n)

Base.show(io::IO, g::ScaledIdentityGate{T,N}) where {T,N} = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityGate of length ", Int(N / 2)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityGate{T,N}) where {T,N} = print(io, ifelse(true == data(g), "", string(data(g), "*")), string("IdentityGate of length ", Int(N / 2)))

struct GenericSquareGate{T,N} <: AbstractSquareGate{T,N}
    data::Array{T,N}
    ishermitian::Bool
    isunitary::Bool
    function GenericSquareGate(data::AbstractArray{T,N}) where {T,N}
        @assert iseven(N) "Gate should be square"
        sg = size(data)
        l = Int(N / 2)
        D = prod(sg[1:l])
        mat = reshape(data, D, D)
        new{T,N}(data, ishermitian(mat), isunitary(mat))
    end
end

abstract type AbstractSite{T,N} <: AbstractArray{T,N} end
abstract type AbstractCenterSite{T} <: AbstractSite{T,3} end
abstract type AbstractVirtualSite{T} <: AbstractSite{T,2} end

struct LinkSite{T} <: AbstractVirtualSite{T}
    Λ::Diagonal{T,Vector{T}}
end
LinkSite(v::Vector) = LinkSite(Diagonal(v))
Base.similar(site::LinkSite) = LinkSite(similar(site.Λ))

struct VirtualSite{T} <: AbstractVirtualSite{T}
    Λ::Matrix{T}
end
Base.similar(site::VirtualSite) = VirtualSite(similar(site.Λ))

struct GenericSite{T} <: AbstractCenterSite{T}
    Γ::Array{T,3}
    purification::Bool
end
Base.similar(site::GenericSite) = GenericSite(similar(site.Γ), site.purification)
Base.setindex!(site::GenericSite, v, I::Vararg{Integer,3}) = (data(site)[I...] = v)

struct OrthogonalLinkSite{T} <: AbstractCenterSite{T}
    Γ::GenericSite{T}
    Λ1::LinkSite{T}
    Λ2::LinkSite{T}
    function OrthogonalLinkSite(Λ1::LinkSite, Γ::GenericSite{T}, Λ2::LinkSite; check = false) where {T}
        if check
            @assert isleftcanonical(Λ1 * Γ) "Error in constructing OrthogonalLinkSite: Is not left canonical"
            @assert isrightcanonical(Γ * Λ2) "Error in constructing OrthogonalLinkSite: Is not right canonical"
            @assert norm(Λ1) ≈ 1
            @assert norm(Λ2) ≈ 1
        end
        new{T}(Γ, Λ1, Λ2)
    end
end
Base.similar(site::OrthogonalLinkSite) = OrthogonalLinkSite(similar(site.Λ1), similar(site.Γ), similar(site.Λ2), check = false)

abstract type AbstractMPS{T<:AbstractCenterSite} <: AbstractVector{T} end

truncation(mps::AbstractMPS) = mps.truncation

mutable struct OpenMPS{T} <: AbstractMPS{OrthogonalLinkSite{T}}
    #In gamma-lambda notation
    Γ::Vector{GenericSite{T}}
    Λ::Vector{LinkSite{T}}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64
    function OpenMPS(
        Γ::Vector{GenericSite{T}},
        Λ::Vector{LinkSite{T}};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION, error = 0.0) where {T}
        new{T}(Γ, Λ, truncation, error)
    end
end


mutable struct LCROpenMPS{T} <: AbstractMPS{GenericSite{T}}
    Γ::Vector{GenericSite{T}}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64

    center::Int
    function LCROpenMPS{T}(
        Γ::Vector{GenericSite{K}};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
        error = 0.0
    ) where {K,T}
        count = 1
        N = length(Γ)
        while count < N + 1 && isleftcanonical(data(Γ[count]))
            count += 1
        end
        center = min(count, N)
        if count < N + 1
            if !(norm(data(Γ[count])) ≈ 1)
                @warn "LCROpenMPS is not normalized.\nnorm= $(norm(data(Γ[count]))))"
            end
            count += 1
        end
        while count < N + 1 && isrightcanonical(data(Γ[count]))
            count += 1
        end
        @assert count == N + 1 "LCROpenMPS is not LR canonical"
        new{T}(Γ, truncation, error, center)
    end
end
function LCROpenMPS(
    Γ::Vector{GenericSite{K}};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    error = 0.0
) where {K}
    LCROpenMPS{K}(Γ; truncation = truncation, error = error)
end
LCROpenMPS(mps::LCROpenMPS) = mps
mutable struct UMPS{T} <: AbstractMPS{OrthogonalLinkSite{T}}
    #In gamma-lambda notation
    Γ::Vector{GenericSite{T}}
    Λ::Vector{LinkSite{T}}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    # Accumulated error
    error::Float64
end

mutable struct CentralUMPS{T} <: AbstractMPS{GenericSite{T}}
    #In gamma-lambda notation
    ΓL::Vector{GenericSite{T}}
    ΓR::Vector{GenericSite{T}}
    Λ::Vector{T}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    # Accumulated error
    error::Float64
end

# numtype(::LCROpenMPS{T}) where {T} = T
# numtype(::UMPS{T}) where {T} = T
# numtype(::CentralUMPS{T}) where {T} = T
# numtype(::OpenMPS{T}) where {T} = T

abstract type BoundaryCondition end
struct OpenBoundary <: BoundaryCondition end
struct InfiniteBoundary <: BoundaryCondition end
#boundaryconditions(::T) where {T<:AbstractMPS} = boundaryconditions(T)
#boundaryconditions(::Type{<:OpenMPS}) = OpenBoundary()
#boundaryconditions(::Type{<:LCROpenMPS}) = OpenBoundary()
#boundaryconditions(::Type{<:UMPS}) = InfiniteBoundary()
boundaryconditions(mps::OpenMPS) = OpenBoundary()
boundaryconditions(mps::LCROpenMPS) = OpenBoundary()
boundaryconditions(mps::UMPS) = InfiniteBoundary()
# mutable struct LROpenMPS{T<:Number} <: AbstractOpenMPS
#     Γ::Vector{AbstractOrthogonalSite}
#     Λ::LinkSite{T}

#     # Max bond dimension and tolerance
#     truncation::TruncationArgs

#     #Accumulated error
#     error::Float64

#     #Orthogonality boundaries
#     center::Int

#     function LROpenMPS(
#         Γ::Vector{AbstractOrthogonalSite},
#         Λ::LinkSite{T};
#         truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
#         center=1, error=0.0,
#     ) where {T}
#         N = length(Γ)
#         @assert 0<center<=N+1 "Error in constructing LROpenMPS: center is not in the chain"
#         @assert norm(data(Λ)) ≈ 1 "Error in constructing LROpenMPS: Singular values not normalized"
#         new{T}(Γ, Λ, truncation, error, center)
#     end
# end

abstract type AbstractEnvironment end
abstract type AbstractInfiniteEnvironment <: AbstractEnvironment end
abstract type AbstractFiniteEnvironment <: AbstractEnvironment end

# struct BoundaryVector{T,N} <: AbstractArray{T,N}
#     data::Array{T,N}
# end
struct BlockBoundaryVector{T,N} #<: AbstractArray{Array{T,N},N}
    data::Array{Array{T,N},N}
    function BlockBoundaryVector(v::Array{T,N}) where {T<:Number,N}
        bv = Array{Array{T,N},N}(undef, (1 for k in 1:N)...)
        bv[1] = v
        return new{T,N}(bv)
    end
    function BlockBoundaryVector(v::Array{Array{T,N},N}) where {T,N}
        #T = promote_type(eltype.(v)...)
        new{T,N}(v)
    end
    # function BlockBoundaryVector(vecs...)
    #     T = promote_type(eltype.(vecs)...)
    #     new{T,length(vecs),typeof(vecs)}(vecs)
    # end
end
# function BlockBoundaryVector(v::Array{T,N}) where {T<:Number,N}
#     bv = Array{Array{T,N},N}(undef, (1 for k in 1:N)...)
#     bv[1] = v
#     return BlockBoundaryVector(bv)
# end

abstract type AbstractTransferMatrix{T,S} end
struct TransferMatrix{F,Fa,T,S} <: AbstractTransferMatrix{T,S}
    f::F
    fa::Fa
    sizes::S
end
TransferMatrix(f::Function, T::DataType, s) = TransferMatrix{typeof(f),Nothing,T,typeof(s)}(f, nothing, s)

struct CompositeTransferMatrix{Maps,T,S} <: AbstractTransferMatrix{T,S}
    maps::Maps
    sizes::S
end
function CompositeTransferMatrix{T}(maps::Maps) where {Maps,T}
    s1 = size(maps[1], 1)
    s2 = size(maps[end], 2)
    CompositeTransferMatrix{Maps,T,typeof(tuple(s1, s2))}(maps, (s1, s2))
end
function CompositeTransferMatrix(map::TransferMatrix{<:Any,<:Any,T,S}) where {T,S}
    CompositeTransferMatrix{typeof(tuple(map)),T,S}(tuple(map), size(map))
end
function CompositeTransferMatrix{T,S}(maps::Maps) where {Maps,T,S}
    s1 = size(maps[1], 1)
    s2 = size(maps[end], 2)
    CompositeTransferMatrix{Maps,T,S}(maps, (s1, s2))
end

abstract type AbstractMPOsite{T} <: AbstractArray{T,4} end

struct MPOsite{T} <: AbstractMPOsite{T}
    data::Array{T,4}
    # ishermitian::Bool
    # isunitary::Bool
end

abstract type AbstractMPO{T<:Number} <: AbstractVector{MPOsite{T}} end

struct MPO{S,T<:Number} <: AbstractMPO{T}
    data::Vector{S}
    boundary::Vector{T}
    function MPO(sites::Vector{S}, boundary::Vector{T}) where {S<:AbstractMPOsite,T}
        K = promote_type(eltype.(sites)..., T)
        new{S,K}(sites, boundary)
    end
end

struct SiteSum{S<:Tuple,T} <: AbstractCenterSite{T}
    sites::S
    function SiteSum(sites::Tuple)
        #println(typeof(Tuple((sites))), promote_rule(eltype.(sites)...))
        new{typeof(sites),promote_type(eltype.(sites)...)}(copy.(sites))
    end
end

struct MPSSum{MPSs<:Tuple,Site<:AbstractSite,Num} <: AbstractMPS{Site}
    states::MPSs
    scalings::Vector{Num}
    function MPSSum(mpss::Tuple, scalings::Vector{Num}) where {Num}
        T = promote_type(numtype.(mpss)...)
        new{typeof(mpss),SiteSum{Tuple{eltype.(mpss)...},T},T}(Tuple(mpss), scalings)
    end
end
function MPSSum(mpss::Tuple)
    MPSSum(mpss, fill(one(promote_type(numtype.(mpss)...)), length(mpss)))
end

struct MPOSiteSum{S<:Tuple,T} <: AbstractMPOsite{T}
    sites::S
    function MPOSiteSum(sites::Tuple)
        new{typeof(sites),promote_type(eltype.(sites)...)}(sites)
    end
end

struct MPOSum{MPOs<:Tuple,Num} <: AbstractMPO{Num}
    mpos::MPOs
    scalings::Vector{Num}
    function MPOSum(mpos::Tuple, scalings::Vector{Num}) where {Num}
        @assert all(length.(mpos) .== length(mpos[1])) "Length of mpos is not the same"
        @assert length(mpos) == length(scalings) "Number of mpos don't match the numver of scale factors."
        new{typeof(mpos),Num}(mpos, scalings)
    end
end

struct ScaledIdentityMPOsite{T} <: AbstractMPOsite{T}
    data::T
    dim::Int
    function ScaledIdentityMPOsite(scaling::T, dim) where {T<:Number}
        new{T}(scaling, dim)
    end
end

struct ScaledIdentityMPO{T} <: AbstractMPO{T}
    data::T
    length::Int
    dim::Int
    function ScaledIdentityMPO(scaling::T, n::Integer, dim) where {T<:Number}
        new{T}(scaling, n, dim)
    end
end

#numtype(ms::Vararg{<:AbstractMPS,Val{1}}) = promote_type(numtype(ms)...)
#numtype(ms::Vararg{<:AbstractMPS,N}) where N = promote_type(numtype.(ms)...)
numtype(::AbstractVector{<:AbstractSite{T}}) where {T} = T
#numtype(::ScaledIdentityMPOsite{T}) where T = T 
numtype(::AbstractMPOsite{T}) where T = T
numtype(::AbstractMPO{T}) where T = T


sites(mps::LCROpenMPS) = mps.Γ
sites(mps::UMPS) = mps.Γ
sites(mps::CentralUMPS) = mps.Γ
sites(mps::OpenMPS) = mps.Γ
