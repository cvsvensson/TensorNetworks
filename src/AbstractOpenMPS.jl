isinfinite(::T) where {T<:AbstractMPS} = isinfinite(boundaryconditions(T))
isinfinite(::OpenBoundary) = false
isinfinite(::InfiniteBoundary) = true

function boundary(::OpenBoundary, mps::AbstractMPS, side::Symbol)
    if side == :right
        return [one(eltype(mps[end]))]
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return [one(eltype(mps[1]))]
    end
end

boundary(::OpenBoundary, cmpss::Tuple, mpss::Tuple, side::Symbol) = kron(
    (conj(boundary(OpenBoundary(), cm, side)) for cm in cmpss)..., (boundary(OpenBoundary(), m, side) for m in mpss)...)

boundary(cmpss::Tuple, mpss::Tuple, args::Vararg) = boundary(boundaryconditions(cmpss[1]),cmpss, mpss, args...)
boundary(mps::AbstractMPS, args::Vararg) = boundary(boundaryconditions(mps),mps, args...)
function boundary(::OpenBoundary,mpo::MPO{T},dir) where T
    D = size(mpo[1],1)
    v = zeros(T,D)
    if dir==:left
        v[1] = one(T)
    elseif dir==:right
        v[D] = one(T)
    else
        @error "Choose direction :left or :right, not $dir"
    end
    return v
end
function boundary(::OpenBoundary,mpo::ScaledIdentityMPO{T},dir) where T
   return [one(T)]
end

function boundary(::InfiniteBoundary, cmps::Tuple, mps::Tuple, side::Symbol)
    _, rhos = transfer_spectrum(cmps, mps, reverse_direction(side), nev = 1)
    return vec(rhos[1])
end
function boundary(::InfiniteBoundary, mps::AbstractMPS, side::Symbol)
    _, rhos = transfer_spectrum(mps, reverse_direction(side), nev = 1)
    return vec(rhos[1])
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
        A, S, B, err = apply_two_site_gate(mps[k], mps[k+1], IdentityGate(Val(2)), truncation)
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

