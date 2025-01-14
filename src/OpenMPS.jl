const DEFAULT_OPEN_DMAX = 20
const DEFAULT_OPEN_TOL = 1e-12
const DEFAULT_OPEN_NORMALIZATION = true
const DEFAULT_OPEN_TRUNCATION = TruncationArgs(
    DEFAULT_OPEN_DMAX,
    DEFAULT_OPEN_TOL,
    DEFAULT_OPEN_NORMALIZATION,
)
Base.getindex(mps::OpenMPS, i::Integer) = OrthogonalLinkSite(mps.Γ[i], mps.Λ[i], mps.Λ[i+1], check = false)

function Base.setindex!(mps::OpenMPS, v::OrthogonalLinkSite{<:Number}, i::Integer)
    mps.Γ[i] = v.Γ
    mps.Λ[i] = v.Λ1
    mps.Λ[i+1] = v.Λ2
    return v
end

#%% Constructors
function OpenMPS(
    Γ::Vector{Array{T,3}},
    Λ::Vector{Vector{T}};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    purification = false,
    error = 0.0
) where {T}
    M = [GenericSite(g, purification) for g in Γ]
    OpenMPS(M, LinkSite.(Λ), truncation = truncation, error = error)
end
function OpenMPS(
    Γ::Vector{GenericSite{T}},
    Λ::Vector{LinkSite{T}},
    mps::OpenMPS;
    error = 0.0
) where {T}
    OpenMPS(Γ, Λ, truncation = mps.truncation, error = mps.error + error)
end

function OpenMPS(
    Γ::Vector{Array{T,3}},
    Λ::Vector{Vector{T}},
    mps::OpenMPS;
    error = 0.0
) where {T}
    OpenMPS(Γ, Λ, truncation = mps.truncation, error = mps.error + error)
end
function OpenMPS(
    M::Vector{GenericSite{T}};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION, error = 0.0
) where {T}
    Γ, Λ, err = to_orthogonal_link(M, truncation)
    OpenMPS(Γ, Λ, truncation = truncation, error = error + err)
end
function OpenMPS(mps::LCROpenMPS)
    OpenMPS(mps[1:end], truncation = mps.truncation, error = mps.error)
end
function OpenMPS(sites::Vector{OrthogonalLinkSite{T}}; truncation, error = 0.0) where {T}
    Γ, Λ = ΓΛ(sites)
    OpenMPS(Γ, Λ, truncation = truncation, error = error)
end

Base.copy(mps::OpenMPS) = OpenMPS(copy(mps.Γ), copy(mps.Λ), mps, error = 0)

"""
    randomOpenMPS(datatype, N, d, D, pur=false, trunc)

Return a random mps
"""
function randomOpenMPS(N, d, D; T = ComplexF64, purification = false,
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION
)
    # Ms = Vector{GenericSite{T}}(undef, N)
    Ms = [randomGenericSite(i == 1 ? 1 : D, d, i == N ? 1 : D, T; purification = purification) for i in 1:N]
    # for i = 1:N
    #     Ms[i] = randomGenericSite( i == 1 ? 1 : D, d, i == N ? 1 : D, T; purification=purification)

    #     #GenericSite(rand(T, i == 1 ? 1 : D, d, i == N ? 1 : D), purification)
    # end
    mps = OpenMPS(Ms, truncation = truncation)
    return mps
end

"""
    identityOpenMPS(N, d; T= ComplexF64, trunc)

Return the identity density matrix as a purification
"""
function identityOpenMPS(N, d; T = ComplexF64,
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION
)
    Γ = Vector{GenericSite{T}}(undef, N)
    Λ = Vector{LinkSite{T}}(undef, N + 1)
    for i = 1:N
        Γ[i] = GenericSite(reshape(Matrix(one(T)I, d, d) / sqrt(d), 1, d^2, 1), true)
        Λ[i] = LinkSite(ones(T, 1))
    end
    Λ[N+1] = LinkSite(ones(T, 1))
    return OpenMPS(Γ, Λ, truncation = truncation)
end

"""
    identityOpenMPS(mps)

Return the identity density matrix as a purification, with similar parameters as the input mps
"""
function identityMPS(mps::OpenMPS)
    N = length(mps)
    d = size(mps[1], 2)
    if ispurification(mps)
        d = Int(sqrt(d))
    end
    trunc = mps.truncation
    return identityOpenMPS(N, d, truncation = trunc)
end

"""
    canonicalize(mps::OpenMPS)

Return the canonical version of the mps
"""
function canonicalize(mps::OpenMPS)
    # M = centralize(mps, 1)
    # canonicalizeM!(M)
    # Γ, Λ, err = ΓΛ_from_M(M, mps.truncation)
    Γ, Λ, err = to_orthogonal_link(RightOrthogonalSite.(mps), mps.truncation)
    return OpenMPS(Γ, Λ, truncation = mps.truncation, error = err)
end

""" 
    canonicalize!(mps::OpenMPS)

Make the mps canonical
"""
function canonicalize!(mps::OpenMPS) #TODO make consistent with LCROpenMPS
    Γ, Λ, err = to_orthogonal_link(RightOrthogonalSite.(mps), mps.truncation)
    mps.Γ = Γ
    mps.Λ = Λ
    mps.error += err
    return
end

"""
to_orthogonal_link_from_right_orth(M::Vector{GenericSite{T}}; trunc::TruncationArgs)

Calculate the ΓΛs from a list of right orthogonal tensors M.
"""
function to_orthogonal_link_from_right_orth(M::Vector{GenericSite{T}}, trunc::TruncationArgs) where {T}
    N = length(M)
    for k in 1:N
        @assert isrightcanonical(M[k]) "Error in to_orthogonal_link: site is not rightorthogonal"
    end
    Γ = Vector{GenericSite{T}}(undef, N)
    Λ = Vector{LinkSite{T}}(undef, N + 1)
    Λ[1] = LinkSite(ones(T, size(M[1], 1)))
    total_error = 0.0
    purification = ispurification(M[1])
    for k = 1:N
        st = size(M[k])
        tensor = reshape(data(M[k]), st[1] * st[2], st[3])
        F = svd(tensor)
        U, S, Vt, D, err = truncate_svd(F, trunc)
        total_error += err
        Γ[k] = inv(Λ[k]) * GenericSite(Array(reshape(U, st[1], st[2], D)), purification)
        Λ[k+1] = LinkSite(S)
        if k < N
            Vt = Λ[k+1] * VirtualSite(Matrix(Vt))
            M[k+1] = Vt * M[k+1]
        end
    end
    return Γ, Λ, total_error
end

"""
to_orthogonal_link(M::Vector{GenericSite{T}}; trunc::TruncationArgs)

Calculate the ΓΛs from a list of tensors M.
"""
function to_orthogonal_link(M::Vector{GenericSite{T}}, trunc::TruncationArgs) where {T}
    MR = to_right_orthogonal(M)[1]
    return to_orthogonal_link_from_right_orth(MR, trunc)
end

function apply_layers(mps::OpenMPS, layers)
    sites, err = apply_layers(mps[1:end], layers, mps.truncation, isperiodic = false)
    return OpenMPS(sites, truncation = mps.truncation, error = mps.error + err)
end

function apply_layers_nonunitary(mps::OpenMPS, layers)
    sites, err = apply_layers_nonunitary(mps[1:end], layers, mps.truncation)
    return OpenMPS(sites, truncation = mps.truncation, error = mps.error + err)
end

set_center!(mps::AbstractVector{<:OrthogonalLinkSite}, ::Integer) = mps
iscenter(mps::AbstractVector{<:OrthogonalLinkSite}, ::Integer) = true

function saveOpenMPS(mps, filename)
    jldopen(filename, "w") do file
        writeOpenMPS(file, mps)
    end
end

function writeOpenMPS(parent, mps)
    write(parent, "Gamma", mps.Γ)
    write(parent, "Lambda", mps.Λ)
    write(parent, "Purification", ispurification(mps))
    write(parent, "Dmax", mps.truncation.Dmax)
    write(parent, "tol", mps.truncation.tol)
    write(parent, "normalize", mps.truncation.normalize)
    write(parent, "error", mps.error)
end

function readOpenMPS(io)
    Γ = read(io, "Gamma")
    Λ = read(io, "Lambda")
    purification = read(io, "Purification")
    Dmax = read(io, "Dmax")
    tol = read(io, "tol")
    normalize = read(io, "normalize")
    error = read(io, "error")
    trunc = TruncationArgs(Dmax, tol, normalize)
    mps = OpenMPS(
        Γ,
        Λ,
        purification = purification,
        truncation = trunc,
        error = error,
    )
    return mps
end

function loadOpenMPS(filename)
    jldopen(filename, "r") do file
        global mps
        mps = readOpenMPS(file)
    end
    return mps
end

entanglement_entropy(mps::OpenMPS) = entanglement_entropy.(mps.Λ)