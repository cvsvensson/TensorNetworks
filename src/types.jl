const BigNumber = Union{ComplexDF64,ComplexDF32,ComplexDF16,Double64,Double32,Double16,BigFloat,Complex{BigFloat}}

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
    function ScaledIdentityGate(scaling::T, ::Val{N}) where {T,N}
        new{T,2 * N}(scaling, isreal(scaling), scaling' * scaling ≈ 1)
    end
end
IdentityGate(n::Integer) = ScaledIdentityGate(true, Val(n))
IdentityGate(::Val{N}) where N = ScaledIdentityGate(true, Val(N))

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
abstract type AbstractPhysicalSite{T} <: AbstractSite{T,3} end
abstract type AbstractVirtualSite{T} <: AbstractSite{T,2} end

struct LinkSite{T} <: AbstractVirtualSite{T}
    Λ::Diagonal{T,Vector{T}}
end
LinkSite(v::Vector) = LinkSite(Diagonal(v))
struct VirtualSite{T} <: AbstractVirtualSite{T}
    Λ::Matrix{T}
end
struct GenericSite{T} <: AbstractPhysicalSite{T}
    Γ::Array{T,3}
    purification::Bool
end

struct OrthogonalLinkSite{T, C, V} <: AbstractPhysicalSite{T}
    Γ::C
    Λ1::V
    Λ2::V
    function OrthogonalLinkSite(Λ1::V, Γ::C, Λ2::V; check = false) where {T}
        if check
            @assert isleftcanonical(Λ1 * Γ) "Error in constructing OrthogonalLinkSite: Is not left canonical"
            @assert isrightcanonical(Γ * Λ2) "Error in constructing OrthogonalLinkSite: Is not right canonical"
            @assert norm(Λ1) ≈ 1
            @assert norm(Λ2) ≈ 1
        end
        new{T,C,V}(Γ, Λ1, Λ2)
    end
end

# struct OrthogonalLinkSite{T} <: AbstractPhysicalSite{T}
#     Γ::GenericSite{T}
#     Λ1::LinkSite{T}
#     Λ2::LinkSite{T}
#     function OrthogonalLinkSite(Λ1::LinkSite, Γ::GenericSite{T}, Λ2::LinkSite; check = false) where {T}
#         if check
#             @assert isleftcanonical(Λ1 * Γ) "Error in constructing OrthogonalLinkSite: Is not left canonical"
#             @assert isrightcanonical(Γ * Λ2) "Error in constructing OrthogonalLinkSite: Is not right canonical"
#             @assert norm(Λ1) ≈ 1
#             @assert norm(Λ2) ≈ 1
#         end
#         new{T}(Γ, Λ1, Λ2)
#     end
# end

abstract type AbstractMPS{T<:AbstractPhysicalSite} <: AbstractVector{T} end

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
        @assert count == N + 1 "LCROpenMPS is not LR canonical. $count != $(N+1)"
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
numtype(ms::Vararg{AbstractVector{<:AbstractSite},<:Any}) = promote_type(numtype.(ms)...)
numtype(::AbstractVector{<:AbstractSite{T}}) where {T} = T
sites(mps::LCROpenMPS) = mps.Γ
sites(mps::UMPS) = mps.Γ
sites(mps::CentralUMPS) = mps.Γ
sites(mps::OpenMPS) = mps.Γ

abstract type BoundaryCondition end
struct OpenBoundary <: BoundaryCondition end
struct InfiniteBoundary <: BoundaryCondition end
boundaryconditions(::T) where {T<:AbstractMPS} = boundaryconditions(T)
boundaryconditions(::Type{<:OpenMPS}) = OpenBoundary()
boundaryconditions(::Type{<:LCROpenMPS}) = OpenBoundary()
boundaryconditions(::Type{<:UMPS}) = InfiniteBoundary()
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
