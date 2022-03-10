isinfinite(mps::AbstractMPS) = isinfinite(boundaryconditions(mps))
isinfinite(::OpenBoundary) = false
isinfinite(::InfiniteBoundary) = true

function boundary(::OpenBoundary, mps::AbstractMPS{<:GenericSite}, side::Symbol)
    if side == :right
        return BoundaryVector([one(eltype(mps[end]))])
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return BoundaryVector([one(eltype(mps[1]))])
    end
end
boundary(bc::OpenBoundary, mps1::AbstractMPS, mps2::AbstractMPS, side) = BlockBoundaryVector(boundary(bc, mps1, side), boundary(bc, mps2, side))
boundary(bc::OpenBoundary, mps::AbstractMPS, mpo::AbstractMPO, side) = boundary(bc, mps, mpo, mps, side)
boundary(bc::OpenBoundary, mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, side) = BlockBoundaryVector(
boundary(bc, mps1, side),boundary(bc, mpo, side), boundary(bc, mps2, side))

boundary(mps::AbstractMPS, args::Vararg) = boundary(boundaryconditions(mps), mps, args...)

function boundary(::InfiniteBoundary, mps::AbstractMPS, g::ScaledIdentityMPO, mps2::AbstractMPS, side::Symbol)
    _, rhos = transfer_spectrum(mps, mps2, reverse_direction(side), nev = 1)
    return BoundaryVector((data(g) ≈ 1 ? 1 : 0) * rhos[1])
end
function boundary(::InfiniteBoundary, mps::AbstractMPS, side::Symbol)
    _, rhos = transfer_spectrum(mps, reverse_direction(side), nev = 1)
    return BoundaryVector(canonicalize_eigenoperator(rhos[1]))
end
function boundary(::InfiniteBoundary, mps::AbstractMPS, mps2::AbstractMPS, side::Symbol)
    _, rhos = transfer_spectrum(mps, mps2, reverse_direction(side), nev = 1)
    return BoundaryVector(canonicalize_eigenoperator(rhos[1]))
end

boundaryvec(args...) = copy(vec(boundary(args...)))


function expectation_value(mps::AbstractMPS{GenericSite}, op, site::Integer)
    mps = set_center(mps, site)
    return expectation_value(mps, op, site, iscanonical = true)
end

function apply_identity_layer(::OpenBoundary, mpsin::AbstractMPS{GenericSite}; kwargs...)
    truncation = get(kwargs, :truncation, mpsin.truncation)
    mps = set_center(mpsin, 1)
    for k in 1:length(mps)-1
        A, S, B, err = apply_two_site_gate(mps[k], mps[k+1], IdentityGate(2), truncation)
        mps.center += 1
        mps[k] = A
        mps[k+1] = S * B
        mps.error += err
    end
    return mps
end

function entanglement_entropy(mpsin::AbstractMPS{GenericSite}, link::Integer)
    N = length(mpsin)
    @assert 0 < link < N
    mps = set_center(mpsin, link)
    _, S, _ = svd(mps[link], :leftorthogonal)
    return entropy(S)
end

