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

struct SquareGate{T,N} <: AbstractSquareGate{T,N}
    data::Array{T,N}
    ishermitian::Bool
    isunitary::Bool
    function SquareGate(data::AbstractArray{T,N}) where {T,N}
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

struct VirtualSite{T,S<:AbstractArray{T,2}} <: AbstractVirtualSite{T}
    Λ::S
end
VirtualSite(site::S) where S = VirtualSite{eltype(S),S}(site)
const LinkSite{T} = VirtualSite{T,Diagonal{T,Vector{T}}}
LinkSite(v::Vector) = VirtualSite(Diagonal(v))
LinkSite(v::Diagonal) = VirtualSite(v)


struct PhysicalSite{T,S<:AbstractArray{T,3}} <: AbstractPhysicalSite{T}
    Γ::S
    purification::Bool
end
PhysicalSite(site::S,pur::Bool = false) where S = PhysicalSite{eltype(S),S}(site,pur)
const DensePSite{T} = PhysicalSite{T,Array{T,3}}

Base.promote_rule(::Type{VirtualSite{T1,S1}}, ::Type{VirtualSite{T2,S2}}) where {T1,T2,S1,S2} =
    VirtualSite{promote_type(T1,T2),promote_type(S1,S2)}
Base.promote_rule(::Type{PhysicalSite{T1,S1}}, ::Type{PhysicalSite{T2,S2}}) where {T1,T2,S1,S2} =
    PhysicalSite{promote_type(T1,T2),promote_type(S1,S2)}

abstract type AbstractPVSite{T,P,V} <: AbstractPhysicalSite{T} end

struct PVSite{T, P, V} <: AbstractPVSite{T,P,V}
    Γ::P
    Λ1::V
    Λ2::V
    function PVSite(Λ1::V1, Γ::P, Λ2::V2; check = false) where {P<:AbstractPhysicalSite,V1<:AbstractVirtualSite,V2<:AbstractVirtualSite}
        T = promote_type(eltype(V1),eltype(V2),eltype(P))
        V = promote_type(V1,V2)
        if check
            @assert isleftcanonical(Λ1 * Γ) "Error in constructing OrthogonalLinkSite: Is not left canonical"
            @assert isrightcanonical(Γ * Λ2) "Error in constructing OrthogonalLinkSite: Is not right canonical"
            @assert norm(Λ1) ≈ 1
            @assert norm(Λ2) ≈ 1
        end
        new{T,P,V}(Γ, Λ1, Λ2)
    end
end

Base.convert(::Type{PhysicalSite{T,S}}, s::PhysicalSite) where {T,S} = PhysicalSite(S(s.Γ),s.purification)

const DensePVSite{T} = PVSite{T,DensePSite{T},LinkSite{T}}
PhysicalSite(site::PVSite) = site.Γ
function VirtualSite(site::PVSite,dir)
    if dir == :left
        return site.Λ1
    else
        @assert dir == :right
        return site.Λ2
    end
end
function PhysicalSite(site::PVSite, dir::Symbol)
    if dir == :left
        return site.Λ1 * site.Γ
    elseif dir == :right
        return site.Γ * site.Λ2
    end
end
function ΓΛ(sites::Vector{<:PVSite})
    Γ = [PhysicalSite(site) for site in sites]
    Λ = [VirtualSite(site,:left) for site in sites]
    push!(Λ,VirtualSite(sites[end],:right))
    return Γ, Λ
end

abstract type AbstractMPS{P<:AbstractPhysicalSite} <: AbstractVector{P} end
abstract type AbstractPVMPS{PV<:AbstractPVSite} <: AbstractMPS{PV} end
abstract type AbstractPMPS{P<:AbstractPhysicalSite} <: AbstractMPS{P} end

mutable struct OpenPVMPS{T,P,V} <: AbstractPVMPS{PVSite{T}}
    #In gamma-lambda notation
    Γ::Vector{P}
    Λ::Vector{V}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64
    function OpenPVMPS(
        Γ::Vector{P},
        Λ::Vector{V};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION, error = 0.0) where {P,V}
        @assert length(Γ) + 1 == length(Λ)
        new{promote_type(eltype(P),eltype(V)),P,V}(Γ, Λ, truncation, error)
    end
end


mutable struct OpenPMPS{T,P} <: AbstractPMPS{P}
    Γ::Vector{P}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64

    center::Int
    function OpenPMPS{T,P}(
        Γ::Vector{<:AbstractPhysicalSite};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
        error = 0.0
    ) where {P,T}
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
        new{T,P}(Γ, truncation, error, center)
    end
end
function OpenPMPS(
    Γ::Vector{P};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    error = 0.0
) where {P}
    OpenPMPS{eltype(P),P}(Γ, truncation = truncation,error = error)
end
# function LCROpenMPS(
#     Γ::Vector{PhysicalSite{K}};
#     truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
#     error = 0.0
# ) where {K}
#     LCROpenMPS{K}(Γ; truncation = truncation, error = error)
# end
mutable struct UMPS{T,P,V} <: AbstractPVMPS{PVSite{T,P,V}}
    #In gamma-lambda notation
    Γ::Vector{P}
    Λ::Vector{V}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    # Accumulated error
    error::Float64
    function UMPS(Γ::Vector{P}, Λ::Vector{V}; truncation::TruncationArgs = DEFAULT_UMPS_TRUNCATION, error = 0.0) where {P,V}
        T = promote_type(eltype(P),eltype(V))
        new{T,P,V}(Γ, Λ, truncation, error)
    end
end

# numtype(::LCROpenMPS{T}) where {T} = T
# numtype(::UMPS{T}) where {T} = T
# numtype(::CentralUMPS{T}) where {T} = T
# numtype(::OpenMPS{T}) where {T} = T
numtype(ms::Vararg{AbstractVector{<:AbstractSite},<:Any}) = promote_type(numtype.(ms)...)
numtype(::AbstractVector{<:AbstractSite{T}}) where {T} = T
sites(mps::OpenPMPS) = mps.Γ
sites(mps::UMPS) = mps.Γ
#sites(mps::CentralUMPS) = mps.Γ
#sites(mps::OpenPVMPS) = mps.Γ

abstract type BoundaryCondition end
struct OpenBoundary <: BoundaryCondition end
struct InfiniteBoundary <: BoundaryCondition end
boundaryconditions(::T) where {T<:AbstractMPS} = boundaryconditions(T)
boundaryconditions(::Type{<:OpenPVMPS}) = OpenBoundary()
boundaryconditions(::Type{<:OpenPMPS}) = OpenBoundary()
boundaryconditions(::Type{<:UMPS}) = InfiniteBoundary()
