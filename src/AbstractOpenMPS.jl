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

function boundary(csites::Tuple, s::Tuple, side::Symbol)
    csites2 = foldl(_split_tuple_c, states.(csites), init=())
    sites2 = foldr(_split_tuple, states.(s), init=())
    cscale = foldl(_split_tuple_c, scaling.(csites), init=())
    scale = foldr(_split_tuple, scaling.(s), init=())
    println.(typeof.(cscale))
    println("_")
    println.(typeof.(scale))
    boundary(csites2, cscale, sites2, scale, side)
end
_split_lazy_v(s, ss::Tuple) = (s, ss...)
_split_lazy_v(ss::Tuple, s) = (ss..., s)
_split_lazy_v(s::Vector{<:LazyProduct}, ss::Tuple) = ([lp.mpos for lp in s], s.mps, ss...)
_split_lazy_v(ss::Tuple, s::Vector{<:LazyProduct}) = (ss..., s.mps, s.mpos...)
#::NTuple{N4,<:Any}
function boundary(csites::Tuple, cscale::Tuple, s::Tuple, scale::Tuple, side::Symbol) #where {N1,N2,N3,N4}

    # println(scale)
    # println(cscale)
    # println.(typeof.(sites2))
    stateitr = Base.product(Base.product(csites...), Base.product(s...))
    scaleitr = Base.product(Base.product(cscale...), Base.product(scale...))
    # println(typeof(collect(scaleitr)))
    # println(size(scaleitr))
    # println(collect(scaleitr))
    # println(collect(Base.product(scaling.(s)...)))
    # println(scaling.(csites))
    #println(typeof(Base.product(s...)))
    println.(typeof.(s))
    T = promote_type(map(s -> eltype(s[1][1]), csites)..., map(s -> eltype(s[1][1]), s)...)
    vecs = [prod(cscale) * prod(scale) * boundary_dense(cs, ss, side) for ((cs, ss), (cscale, scale)) in zip(stateitr, scaleitr)]
    BlockBoundaryVector(vecs)::BlockBoundaryVector{T,length(csites) + length(s)}
end
_split_tuple_c(t::Tuple, mp) = (t..., mp)
_split_tuple_c(t::Tuple, mp::Tuple) = (t..., reverse(mp)...)
_split_tuple(mp, t::Tuple) = (mp, t...)
_split_tuple(mp::Tuple, t::Tuple) = (mp..., t...)


states(mps::MPSSum) = mps.states

function states(mps::MPSSum)
    [mp for mp in mps]

end
states(mpo::MPOSum) = mpo.mpos
states(mps::Union{AbstractMPS,AbstractMPO}) = [mps]
states(mps::LazyProduct) = (states.(mps.mpos)..., states(mps.mps)) #(map(mpo -> [states(mpo)], mps.mpos)..., [states(mps.mps)])
scaling(mps::MPSSum) = mps.scalings
scaling(mpo::MPOSum) = mpo.scalings
scaling(mps::Union{AbstractMPS,AbstractMPO}) = [one(numtype(mps))]
scaling(mps::LazyProduct) = (scaling.(mps.mpos)..., scaling(mps.mps)) #(map(mpo -> [scaling(mpo)], mps.mpos)..., [scaling(mps.mps)])
statesscaling(mps::MPSSum) = zip(mps.states, mps.scalings)
statesscaling(mpo::MPOSum) = zip(mpo.mpos, mpo.scalings)
statesscaling(mps::Union{AbstractMPS,AbstractMPO}) = [(mps, one(numtype(mps)))]


function boundary_dense(csites::Tuple, sites::Tuple, side::Symbol)
    K = promote_type(numtype.(sites)...)
    # println.(numtype.(sites))
    @assert (K) <: Number "Error: K is not a number, $K"
    newcsites3 = foldl(_split_lazy, csites, init=())
    newsites3 = foldr(_split_lazy, sites, init=())
    # cscale, newcsites3 = foldl(_remove_identity, newcsites2, init=(one(K), ()))
    # scale, newsites3 = foldr(_remove_identity, newsites2, init=(one(K), ()))
    #println.(typeof.(newcsites3))
    # final_scale = side==:left ? (scale * cscale) : one(K)
    T = boundary(boundaryconditions(csites[1]), newcsites3, newsites3, side)
    return T#::Union{Array{K,<:Any},BlockBoundaryVector{K,<:Any,<:Any}}
end

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

