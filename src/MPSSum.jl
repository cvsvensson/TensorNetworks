
struct MPSSum{MPS<:AbstractMPS,Site<:AbstractSite,Num} <: AbstractMPS{Site}
    states::Vector{MPS}
    scalings::Vector{Num}
    function MPSSum(mpss::Vector{T}) where {T<:AbstractMPS}
        new{T,SiteSum{eltype(mpss[1]),numtype(mpss...)},numtype(mpss...)}([mpss...],fill(1,length(mpss)))
    end
    function MPSSum(mpss::Vector{T},scalings::Vector{Num}) where {T<:AbstractMPS,Num}
        new{T,SiteSum{eltype(mpss[1]), numtype(mpss...)},numtype(mpss...)}(mpss,scalings)
    end
end
Base.show(io::IO, mps::MPSSum) =
    (print(io, "MPS: ", typeof(mps), "\nSites: ", eltype(mps) ,"\nLength: ", length(mps), "\nSum of ", length(mps.states) ," mps's\nWith scalings "); show(io,mps.scalings))
Base.show(io::IO, m::MIME"text/plain", mps::MPSSum) = show(io,mps)
Base.size(mps::MPSSum) = (length(mps), length(mps.states))
Base.length(mps::MPSSum) = length(mps.states[1])
struct SiteSum{S<:AbstractSite,T} <: AbstractCenterSite{T}
    sites::Vector{S}
    function SiteSum(sites::Vector{S}) where {S<:AbstractCenterSite}
        new{S,eltype(sites[1])}(sites)
    end 
end
SiteSum(site::AbstractCenterSite) = SiteSum([site])
Base.show(io::IO, mps::SiteSum) =
    print(io, "SiteSum: ", typeof(mps), "\nSites: ", eltype(mps) ,"\nLength: ", length(mps.sites))
Base.show(io::IO, m::MIME"text/plain", mps::SiteSum) = show(io,mps)

Base.size(sites::SiteSum) = (sum(size.(sites.sites,1)),size(sites.sites[1],2),sum(size.(sites.sites,3)))

Base.:+(mps1::AbstractMPS,mps2::AbstractMPS) = MPSSum([mps1, mps2])
Base.:+(mps::AbstractMPS,sum::MPSSum) = MPSSum(1*mps, sum)
Base.:+(sum::MPSSum,mps::AbstractMPS) = MPSSum(sum, 1*mps)
Base.:+(s1::MPSSum,s2::MPSSum) = MPSSum(vcat(s1.states,s2.states),vcat(s1.scalings,s2.scalings))
Base.:*(x::Number, mps::AbstractMPS) = MPSSum([mps],[x])
Base.:*(mps::AbstractMPS,x::Number) = MPSSum([mps],[x])
Base.:*(x::Number, mps::MPSSum) = MPSSum(mps.states,x*mps.scalings)
Base.:*(mps::MPSSum, x::Number) = MPSSum(mps.states,x*mps.scalings)
Base.:-(mps::AbstractMPS) = (-1)*mps
Base.:-(mps::AbstractMPS,mps2::AbstractMPS) = mps + (-1)*mps2

Base.IndexStyle(::Type{<:MPSSum}) = IndexLinear()
Base.getindex(sum::MPSSum,i::Integer) = SiteSum([state[i] for state in sum.states])


Base.getindex(sum::SiteSum,i::Integer) = sum.sites[i]
Base.IndexStyle(::Type{<:SiteSum}) = IndexLinear()
reverse_direction(sitesum::SiteSum) = SiteSum(reverse_direction.(sitesum.sites))
ispurification(sitesum::SiteSum) = ispurification(sitesum[1])

function _transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate, Γ2::Vector{<:SiteSum}) 
    #FIXME Might screw up type stability, so might need a TransferMatrix struct?
    N1 = length(Γ1)
    N2 = length(Γ2)
    Ts = [_transfer_right_gate(getindex.(Γ1,n1),gate,getindex.(Γ2,n2)) for n1 in 1:N1, n2 in 1:N2]
    DL = sum(size.(Ts,1))
    DR = sum(size.(Ts,2))
    function f(v)
        [Ts[n1,n2] * v[n1,n2,:] for n1 in 1:N1, n2 in 1:N2]
    end
    function f_adjoint(v)
        [Ts[n1,n2]' * v[n1,n2,:] for n1 in 1:N1, n2 in 1:N2]
    end
    T = promote_type(eltype.(Γ2)...)
    return LinearMap{T}(f,f_adjoint,DL,DR)
end
_transfer_right_gate(Γ1::Vector{<:AbstractSite}, gate::GenericSquareGate, Γ2::SiteSum) = _transfer_right_gate(SiteSum.(Γ1),gate, Γ2)
_transfer_right_gate(Γ1::Vector{<:SiteSum}, gate::GenericSquareGate, Γ2::Vector{<:AbstractSite}) = _transfer_right_gate(Γ1,gate, SiteSum.(Γ2))
