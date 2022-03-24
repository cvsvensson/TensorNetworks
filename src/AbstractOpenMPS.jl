isinfinite(mps::AbstractMPS) = isinfinite(boundaryconditions(mps))
isinfinite(::OpenBoundary) = false
isinfinite(::InfiniteBoundary) = true

function boundary(::OpenBoundary, mps::AbstractMPS{<:Union{<:GenericSite,<:OrthogonalLinkSite}}, side::Symbol)
    if side == :right
        return [one(eltype(mps[end]))]
    else
        if side !== :left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return [one(eltype(mps[1]))]
    end
end

# boundary(bc::OpenBoundary, mps1::AbstractMPS, mps2::AbstractMPS, side) = tensor_product(boundary(bc, mps1, side), boundary(bc, mps2, side))
# boundary(bc::OpenBoundary, mps::AbstractMPS, mpo::AbstractMPO, side) = boundary(bc, mps, mpo, mps, side)
# boundary(bc::OpenBoundary, mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, side) = tensor_product(
#     (boundary(bc, m, side) for m in (mps1, mpo, mps2))...)

# boundary(mps::AbstractMPS, args::Vararg) = boundary(boundaryconditions(mps), mps, args...)

boundary(::OpenBoundary, cmpss::Tuple, mpss::Tuple, side::Symbol) = tensor_product(
    (conj(boundary(OpenBoundary(), cm, side)) for cm in cmpss)..., (boundary(OpenBoundary(), m, side) for m in mpss)...)

boundary(::OpenBoundary, lp::LazyProduct, side::Symbol) = tensor_product((boundary(OpenBoundary(), mp, side) for mp in (lp.mpos..., lp.mps))...)

# function boundary(csites::NTuple{<:Any,Union{<:AbstractMPS,MPSSum,MPOSum}}, sites::NTuple{<:Any,Union{<:AbstractMPS,MPSSum,MPOSum}}, side::Symbol)

#     BlockBoundaryVector([boundary(cs, ss,site) for (cs, ss) in Base.product(Base.product(states.(csites)...), Base.product(states.(s)...))])
#     if side == :right
#         return tensor_product(BlockBoundaryVector.([boundary(OpenBoundary(), mps.states[k], :right) for k in 1:length(mps.states)...])...)::BlockBoundaryVector
#     else
#         if side !== :left
#             @warn "No direction chosen for the boundary vector. Defaulting to :left"
#         end
#         return tensor_product(BlockBoundaryVector.([mps.scalings[k] .* boundary(OpenBoundary(), mps.states[k], :left) for k in 1:length(mps.states)])...)::BlockBoundaryVector
#     end
# end
function boundary(csites::Tuple, s::Tuple, side::Symbol)
    stateitr = Base.product(Base.product(states.(csites)...), Base.product(states.(s)...))
    scaleitr = Base.product(Base.product(scaling.(csites)...), Base.product(scaling.(s)...))
    vecs = [prod(cscale) * prod(scale) * boundary_dense(cs, ss, side) for ((cs, ss), (cscale, scale)) in zip(stateitr, scaleitr)]
    BlockBoundaryVector(vecs)
end
states(mps::MPSSum) = mps.states

# function states(mps::MPSSum)
#     [mp for mp in mps]

# end
states(mpo::MPOSum) = mpo.mpos
states(mps::Union{AbstractMPS,AbstractMPO}) = [mps]
states(mps::LazyProduct) = [mps]#(states.(mps.mpos)..., states(mps.mps))
scaling(mps::MPSSum) = mps.scalings
scaling(mpo::MPOSum) = mpo.scalings
scaling(mps::Union{AbstractMPS,AbstractMPO}) = [one(numtype(mps))]
scaling(mps::LazyProduct) = [one(numtype(mps))] #[(scaling.(mps.mpos)..., scaling(mps.mps))
statesscaling(mps::MPSSum) = zip(mps.states, mps.scalings)
statesscaling(mpo::MPOSum) = zip(mpo.mpos, mpo.scalings)
statesscaling(mps::Union{AbstractMPS,AbstractMPO}) = [(mps, one(numtype(mps)))]

# function _filter_sites(csites::Tuple, sites::Tuple)


# end

function boundary_dense(csites::Tuple, sites::Tuple, side::Symbol)
    K = promote_type(numtype.(sites)...)
    # println.(numtype.(sites))
    @assert (K) <: Number "Error: K is not a number, $K"
    newcsites2 = foldl(_split_lazy, csites, init=())
    newsites2 = foldr(_split_lazy, sites, init=())
    cscale, newcsites3 = foldl(_remove_identity, newcsites2, init=(one(K), ()))
    scale, newsites3 = foldr(_remove_identity, newsites2, init=(one(K), ()))
    #println.(typeof.(newcsites3))
    final_scale = side==:left ? (scale * cscale) : one(K)
    T = final_scale*boundary(boundaryconditions(csites[1]), newcsites3, newsites3, side)
    return T#::Union{Array{K,<:Any},BlockBoundaryVector{K,<:Any,<:Any}}
end
# function boundary(csites::Tuple, sites::Tuple, side::Symbol)
#     K = promote_type(numtype.(sites)..., numtype.(csites)...)
#     newcsites2 = foldl(_split_lazy, csites, init=())
#     newsites2 = foldr(_split_lazy, sites, init=())
#     cscale, newcsites3 = foldl(_remove_identity, newcsites2, init=(one(K), ()))
#     scale, newsites3 = foldr(_remove_identity, newsites2, init=(one(K), ()))
#     #println.(typeof.(newcsites3))
#     T = (scale * cscale) * boundary(boundaryconditions(csites[1]), newcsites3, newsites3, side)
#     return T#::Union{Array{K,<:Any},BlockBoundaryVector{K,<:Any,<:Any}}
# end

# function boundary(::InfiniteBoundary, mps::AbstractMPS, g::ScaledIdentityMPO, mps2::AbstractMPS, side::Symbol)
#     _, rhos = transfer_spectrum((mps,), (mps2,), reverse_direction(side), nev = 1)
#     return (data(g) ≈ 1 ? 1 : 0) * rhos[1]
# end
function boundary(::InfiniteBoundary, cmpss::Tuple, mpss::Tuple, side::Symbol)
    _, rhos = transfer_spectrum(cmpss, mpss, reverse_direction(side), nev=1)
    return canonicalize_eigenoperator(rhos[1])
end

# function boundary(::InfiniteBoundary, mps::AbstractMPS, side::Symbol)
#     _, rhos = transfer_spectrum((mps,), (mps,), reverse_direction(side), nev = 1)
#     return canonicalize_eigenoperator(rhos[1])
# end
# function boundary(::InfiniteBoundary, mps::AbstractMPS, mps2::AbstractMPS, side::Symbol)
#     _, rhos = transfer_spectrum((mps,), (mps2,), reverse_direction(side), nev = 1)
#     return canonicalize_eigenoperator(rhos[1])
# end

#boundaryvec(args...) = copy(vec(boundary(args...)))

function expectation_value(mps::AbstractMPS{GenericSite}, op, site::Integer)
    mps = set_center(mps, site)
    return expectation_value(mps, op, site, iscanonical=true)
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

