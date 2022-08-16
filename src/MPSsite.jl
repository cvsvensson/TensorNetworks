Base.eltype(::AbstractSite{T}) where T = T

Base.permutedims(site::PhysicalSite, perm) = PhysicalSite(permutedims(site.Γ, perm), site.purification)
Base.copy(site::PVSite) = PVSite(copy(site.Λ1), copy(site.Γ), copy(site.Λ2))
Base.copy(site::PhysicalSite) = PhysicalSite([copy(getfield(site, k)) for k = 1:length(fieldnames(PhysicalSite))]...)
Base.copy(site::LinkSite) = LinkSite(data(site))

Base.@propagate_inbounds Base.getindex(site::AbstractSite, I...) = getindex(data(site), I...)

Base.size(site::AbstractSite, dim) = size(data(site), dim)
Base.size(site::AbstractSite) = size(data(site))
Base.size(site::PVSite) = size(site.Γ)
Base.size(site::PVSite, dim) = size(site.Γ, dim)
Base.size(site::VirtualSite, dim) = size(site.Λ, dim)
Base.size(site::VirtualSite) = size(site.Λ)
Base.length(site::LinkSite) = length(site.Λ)

Base.conj(s::PhysicalSite) = PhysicalSite(conj(data(s)), ispurification(s))
Base.conj(s::PVSite) = PVSite(s.Λ1, conj(s.Γ), s.Λ2)

Base.:*(s1::PhysicalSite, s2::PhysicalSite) = PhysicalSite(coarse_grain((@tensor s12[:] := data(s1)[-1,-2,1]*data(s2)[1,-3,-4]),2),ispurification(s1))

function coarse_grain(A::AbstractArray{T,N},n::I) where {N,T,I<:Integer}
    sA = size(A)
    sA2::NTuple{N-1,I} = (sA[1:n-1]...,sA[n]*sA[n+1],sA[n+2:end]...)
    reshape(A,sA2)
end

#Base.isapprox(s1::AbstractSite,s2::AbstractSite) = isapprox(data(s1),data(s2))
Base.isapprox(s1::PVSite, s2::PVSite) = isapprox(s1.Γ, s2.Γ) && isapprox(s1.Λ1, s2.Λ1) && isapprox(s1.Λ2, s2.Λ2)
ispurification(site::PhysicalSite) = site.purification
ispurification(site::PVSite) = ispurification(site.Γ)
ispurification(::VirtualSite) = false

LinearAlgebra.lmul!(a::Number,B::PhysicalSite) = PhysicalSite(lmul!(a,B.Γ), B.purification)

link(site::PhysicalSite, dir) = I
function link(site::PVSite, dir)
    if dir == :left
        site.Λ1
    elseif dir == :right
        site.Λ2
    else
        error("Choose direction :left or :right")
    end
end
data(site::PhysicalSite) = site.Γ
data(site::VirtualSite) = site.Λ
data(site::LinkSite) = site.Λ
data(site::PhysicalSite, dir) = site.Γ
data(site::PVSite, dir) = data(PhysicalSite(site, dir))
data(site::VirtualSite, dir) = site.Λ
data(site::LinkSite, dir) = site.Λ
data(site::PVSite) = data(site.Λ1 * site.Γ * site.Λ2)

MPOsite(site::PhysicalSite) = (s = size(site); MPOsite(reshape(data(site), s[1], s[2], 1, s[3])))
MPOsite(site::PhysicalSite, dir) = MPOsite(site)
MPOsite(site::PVSite, dir) = MPOsite(PhysicalSite(site, dir), dir)

MPOsite{K}(site::PhysicalSite) where {K} = (s = size(site); MPOsite{K}(reshape(data(site), s[1], s[2], 1, s[3])))
MPOsite{K}(site::PhysicalSite, dir) where {K} = MPOsite{K}(site)
MPOsite{K}(site::PVSite, dir) where {K} = MPOsite{K}(PhysicalSite(site, dir), dir)

Base.sqrt(site::LinkSite) = LinkSite(sqrt(data(site)))

isleftcanonical(site::AbstractSite) = isleftcanonical(data(site))
isleftcanonical(site::PVSite) = isleftcanonical(site.Λ1 * site.Γ)
isrightcanonical(site::AbstractSite) = isrightcanonical(data(site))
isrightcanonical(site::PVSite) = isrightcanonical(site.Γ * site.Λ2)
iscanonical(site::PVSite) = isrightcanonical(site) && isleftcanonical(site) && norm(site.Λ1) ≈ 1 && norm(site.Λ2) ≈ 1

entanglement_entropy(Λ::LinkSite) = -sum(data(Λ) * log(data(Λ)))

PhysicalSite(site::PhysicalSite) = site
PhysicalSite(site::PhysicalSite, dir) = site

Base.convert(::Type{LinkSite{T}}, Λ::LinkSite{K}) where {K,T} = LinkSite(Diagonal{T}(data(Λ)))
# Base.convert(::Type{GenericSite{T}}, site::GenericSite{K}) where {K,T} = GenericSite(convert.(T,data(site)),ispurification(site))

LinearAlgebra.norm(site::AbstractSite) = norm(data(site))
function Base.promote_rule(A::Type{<:Diagonal{<:Any,V}}, B::Type{<:Diagonal{<:Any,W}}) where {V,W}
    X = promote_type(V, W)
    T = eltype(X)
    isconcretetype(T) && return Diagonal{T,X}
    return typejoin(A, B)
end

function LinearAlgebra.ishermitian(site::MPOsite)
    ss = size(site)
    if !(ss[1] == 1 && ss[4] == 1)
        return false
    else
        m = reshape(data(site), ss[2], ss[3])
        return ishermitian(m)
    end
end

function LeftOrthogonalSite(site::PVSite; check = true)
    L = site.Λ1 * site.Γ
    check || @assert isleftcanonical(L) "In LeftOrthogonalSite: site is not left canonical"
    return L
end
function RightOrthogonalSite(site::PVSite; check = true)
    R = site.Γ * site.Λ2
    check || @assert isrightcanonical(R) "In RightOrthogonalSite: site is not right canonical"
    return R
end
function OrthogonalSite(site::PVSite, side; check = true)
    if side == :left
        return LeftOrthogonalSite(site, check = check)
    elseif side == :right
        return RightOrthogonalSite(site, check = check)
    else
        "Error in OrthogonalSite: choose side :left or :right"
    end
end

"""
to_left_orthogonal(site::GenericSite, dir)-> A,R,DB

Return the left orthogonal form of the input site as well as the remainder.
"""
function to_left_orthogonal(site::PhysicalSite; full = false, method = :qr, truncation = DEFAULT_OPEN_TRUNCATION)
    D1, d, D2 = size(site)
    M = reshape(data(site), D1 * d, D2)
    if method == :svd
        Usvd, S, Vt, _, _ = split_truncate(M, truncation)
        A = Matrix(Usvd)
        R = Diagonal(S) * Vt
    else
        if method !== :qr
            @warn "Choose :qr or :svd as method in 'to_left_orthogonal', not $method. Defaulting to QR"
        end
        Uqr, R = qr(M)
        A = full ? Uqr * Matrix(I, D2, D2) : Matrix(Uqr)
    end
    Db = size(R, 1) # intermediate bond dimension
    orthSite = PhysicalSite(reshape(A, D1, d, Db), site.purification)
    V = VirtualSite(R)
    return orthSite, V
end
function to_right_orthogonal(site::PhysicalSite; full = false, method = :qr) 
    #FIXME: reverse_direction needs memory allocation. So write a specialized version instead.
    #M = permutedims(site,[3,2,1])
    L, V = to_left_orthogonal(reverse_direction(site), full = full, method = method)
    reverse_direction(L), transpose(V)
end

function LinearAlgebra.svd(site::PhysicalSite, orth = :leftorthogonal)
    s = size(site)
    if orth == :rightorthogonal
        m = reshape(data(site), s[1], s[2] * s[3])
        F = svd(m)
        U = VirtualSite(F.U)
        S = LinkSite(F.S)
        Vt = PhysicalSite(reshape(F.Vt, s), site.purification)
    else
        if orth !== :leftorthogonal
            @warn "Choose :leftorthogonal or :rightorthogonal, not $orth. Defaulting to :leftorthogonal"
        end
        m = reshape(data(site), s[1] * s[2], s[3])
        F = svd(m)
        U = PhysicalSite(reshape(F.U, s), site.purification)
        S = LinkSite(F.S)
        Vt = VirtualSite(F.Vt)
    end
    return U, S, Vt
end

reverse_direction(site::PhysicalSite) = PhysicalSite(permutedims(data(site), [3, 2, 1]), site.purification)
reverse_direction(site::PVSite) = PVSite(link(site, :right), reverse_direction(site.Γ), link(site, :left))

Base.transpose(G::VirtualSite) = VirtualSite(Matrix(transpose(G.Λ)))
Base.transpose(Λ::LinkSite) = Λ

Base.:*(Γ::PhysicalSite, α::Number) = PhysicalSite(α * data(Γ), Γ.purification)
Base.:*(α::Number, Γ::PhysicalSite) = PhysicalSite(α * data(Γ), Γ.purification)
Base.:/(Γ::PhysicalSite, α::Number) = PhysicalSite(data(Γ) / α, Γ.purification)
Base.:*(Λ::LinkSite, G::VirtualSite) = VirtualSite(reshape(diag(Λ.Λ), size(G, 1), 1) .* G.Λ)
Base.:*(G::VirtualSite, Λ::LinkSite) = VirtualSite(reshape(diag(Λ.Λ), 1, size(G, 2)) .* G.Λ)
Base.:/(Γ::LinkSite, α::Number) = LinkSite(data(Γ) / α)
Base.:/(Γ::VirtualSite, α::Number) = VirtualSite(data(Γ) / α)

Base.:*(Λ::LinkSite, α::Number) = LinkSite(α * data(Λ))
Base.:*(α::Number, Λ::LinkSite) = LinkSite(α * data(Λ))

function Base.:*(v::Vector{<:Number}, Γ::PhysicalSite)
    v2 = reshape(v, 1, length(v))
    @tensor new[:] := v2[-1, l] * data(Γ)[l, -2, -3]
    PhysicalSite(new, Γ.purification)
end
function Base.:*(Γ::PhysicalSite, v::Vector{<:Number})
    v2 = reshape(v, length(v), 1)
    @tensor new[:] := data(Γ)[-1, -2, r] * v2[r, -3]
    PhysicalSite(new, Γ.purification)
end

Base.:*(Λ::LinkSite, Γ::PhysicalSite) = PhysicalSite(reshape(diag(Λ.Λ), size(Γ, 1), 1, 1) .* data(Γ), Γ.purification)
Base.:*(Γ::PhysicalSite, Λ::LinkSite) = PhysicalSite(data(Γ) .* reshape(diag(Λ.Λ), 1, 1, size(Γ, 3)), Γ.purification)
function Base.:*(G::VirtualSite, Γ::PhysicalSite)
    sG = size(G)
    sΓ = size(Γ)
    Γnew = reshape(data(G) * reshape(data(Γ), sΓ[1], sΓ[2] * sΓ[3]), sG[1], sΓ[2], sΓ[3])
    # @tensor Γnew[:] := data(G)[-1,1] * data(Γ)[1,-2,-3]
    PhysicalSite(Γnew, Γ.purification)
end
function Base.:*(Γ::PhysicalSite, G::VirtualSite)
    sG = size(G)
    sΓ = size(Γ)
    #FIXME: check whether tullio can be faster or allocate less.
    Γnew = reshape(reshape(data(Γ), sΓ[1] * sΓ[2], sΓ[3]) * data(G), sΓ[1], sΓ[2], sG[2])
    # @tensor Γnew[:] := data(Γ)[-1,-2,1] * data(G)[1,-3]
    PhysicalSite(Γnew, Γ.purification)
end

Base.inv(G::VirtualSite) = VirtualSite(inv(G.Λ))
Base.inv(Λ::LinkSite) = LinkSite(inv(Λ.Λ))

function Base.:*(gate::SquareGate, Γ::Tuple{PhysicalSite,PhysicalSite})
    g = data(gate)
    s1, s2 = size.(Γ)
    m1 = reshape(data(Γ[1]), s1[1] * s1[2], s1[3])
    m2 = reshape(data(Γ[2]), s2[1], s2[2] * s2[3])
    m12 = reshape(m1 * m2, s1[1], s1[2], s2[2], s2[3])
    return @tullio theta[l, lu, ru, r] := m12[l, cl, cr, r] * g[lu, ru, cl, cr]
    #m12 = reshape(permutedims(reshape(m1*m2,s1[1],s1[2],s2[2],s2[3]),[2,3,1,4]), s1[2]*s2[2],s1[1]*s2[3])
    #return permutedims(reshape(reshape(g,sg[1]*sg[2],sg[3]*sg[4]) *m12,s1[2],s2[2],s1[1],s2[3]), [3,1,2,4])
    #@tensoropt (5,-1,-4) theta[:] := L[-1,2,5]*R[5,3,-4]*g[-2,-3,2,3]
end
function Base.:*(gate::ScaledIdentityGate{<:Number,4}, Γ::Tuple{PhysicalSite,PhysicalSite})
    s1, s2 = size.(Γ)
    m1 = reshape(data(Γ[1]), s1[1] * s1[2], s1[3])
    m2 = reshape(data(Γ[2]), s2[1], s2[2] * s2[3])
    return reshape(rmul!(m1 * m2, data(gate)), s1[1], s1[2], s2[2], s2[3]) # Fast at runtime and at compilation
    #return data(gate)*m1*m2
    #return reshape(data(gate)*m1*m2,s1[1],s1[2],s2[2],s2[3])
    #@tensor theta[:] := data(gate)*data(Γ[1])[-1,-2,1]*data(Γ[2])[1,-3,-4]
end

# function Base.:*(gate::AbstractSquareGate, Γ::Tuple{OrthogonalLinkSite, OrthogonalLinkSite})
# 	@assert Γ[1].Λ2 == Γ[2].Λ1 "Error in applying two site gate: The sites do not share a link"
# 	ΓL = LeftOrthogonalSite(Γ1)
# 	ΓR = Γ2.Λ1*RightOrthogonalSite(Γ2)
# 	gate*(ΓL,ΓR)
# end

# PVSite(Γ::PhysicalSite, Λ1::LinkSite, Λ2::LinkSite; check = false) = PVSite(Λ1, Γ, Λ2, check = check)

"""
	compress(Γ1::GenericSite, Γ2::GenericSite, args::TruncationArgs)

Contract and compress the two sites using the svd. Return two U,S,V,err where U is a LeftOrthogonalSite, S is a LinkSite and V is a RightOrthogonalSite
"""
compress(Γ1::AbstractSite, Γ2::AbstractSite, args::TruncationArgs) = apply_two_site_gate(Γ1, Γ2, IdentityGate(Val(2)), args)

"""
	apply_two_site_gate(Γ1::GenericSite, Γ2::GenericSite, gate, args::TruncationArgs)

Contract and compress the two sites using the svd. Return two U,S,V,err where U is a LeftOrthogonalSite, S is a LinkSite and V is a RightOrthogonalSite
"""
function apply_two_site_gate(Γ1::PhysicalSite, Γ2::PhysicalSite, gate, args::TruncationArgs)
    theta = gate * (Γ1, Γ2)
    DL, d, d, DR = size(theta)
    U, S, Vt, Dm, err = split_truncate(reshape(theta, DL * d, d * DR), args)
    U2 = PhysicalSite(Array(reshape(U, DL, d, Dm)), ispurification(Γ1))
    Vt2 = PhysicalSite(Array(reshape(Vt, Dm, d, DR)), ispurification(Γ2))
    S2 = LinkSite(S)
    return U2, S2, Vt2, err
end

function apply_two_site_gate(Γ1::PVSite, Γ2::PVSite, gate, args::TruncationArgs)
    @assert Γ1.Λ2 ≈ Γ2.Λ1 "Error in apply_two_site_gate: The sites do not share a link"
    ΓL = LeftOrthogonalSite(Γ1)
    ΓR = Γ2.Λ1 * RightOrthogonalSite(Γ2)
    U, S, Vt, err = apply_two_site_gate(ΓL, ΓR, gate, args)
    U2 = inv(Γ1.Λ1) * U
    Vt2 = Vt * inv(Γ2.Λ2)
    Γ1new = PVSite(Γ1.Λ1, U2, S)
    Γ2new = PVSite(S, Vt2, Γ2.Λ2)
    return Γ1new, Γ2new, err
end

function to_left_right_orthogonal(M::Vector{<:PhysicalSite{T}}; center = 1, method = :qr) where {T}
    N = length(M)
    @assert N + 1 >= center >= 0 "Error in 'to_left_right_orthogonal': Center is not within the chain, center==$center"
    M = deepcopy(M)
    Γ = similar(M)
    local G::VirtualSite{T}
    for i in 1:center-1
        Γ[i], G = to_left_orthogonal(M[i], method = method)
        i < N && (M[i+1] = G * M[i+1])
    end
    for i in N:-1:center+1
        Γ[i], G = to_right_orthogonal(M[i], method = method)
        i > 1 && (M[i-1] = M[i-1] * G)
    end

    if center == 0 || center == N + 1
        lmul!(sign(data(G)[1]),Γ[end])
        if !(data(G) ≈ ones(T, 1, 1))
            @warn "In to_orthogonal!: remainder is not 1 at the end of the chain. $G"
        end
    else
        Γ[center] = M[center] / norm(M[center])
    end
    return Γ
end
function to_right_orthogonal(M::Vector{<:PhysicalSite}; method = :qr) #where {T,S}
    out = similar(M) #Vector{PhysicalSite{T}}(undef, length(M))
    M2 = copy(M)
    N = length(M)
    local G#::VirtualSite{T}
    for i in N:-1:1
        out[i], G = to_right_orthogonal(M2[i], method = method)
        if i > 1
            M2[i-1] = M2[i-1] * G
        end
    end
    return out, G
end


function randomDensePhysicalSite(Dl, d, Dr, T = ComplexF64; purification = false)
    Γ = rand(T, Dl, d, Dr)
    return PhysicalSite(Γ / norm(Γ), purification)
end

function randomLeftOrthogonalSite(Dl, d, Dr, T = ComplexF64; purification = false, method = :qr)
    Γ = randomDensePhysicalSite(Dl, d, Dr, T, purification = purification)
    return to_left_orthogonal(Γ, method = method)[1]
end

function randomRightOrthogonalSite(Dl, d, Dr, T = ComplexF64; purification = false, method = :qr)
    Γ = randomDensePhysicalSite(Dl, d, Dr, T, purification = purification)
    return to_right_orthogonal(Γ, method = method)[1]
end

function randomOrthogonalLinkSite(Dl, d, Dr, T = ComplexF64; purification = false)
    Γ = randomDensePhysicalSite(Dl, d, Dr, T, purification = purification)
    #_, ΛL0,_ = svd(Γ, :leftorthogonal)
    _, ΛL0, ΓR = svd(Γ, :rightorthogonal)
    ΛL = ΛL0 / norm(ΛL0)
    R = ΛL * ΓR
    U, ΛR, _ = svd(R, :leftorthogonal)
    final = inv(ΛL) * U
    return PVSite(ΛL,final, ΛR)
end
