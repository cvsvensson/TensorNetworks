"""
    DMRG(mpo, mps_input, orth=[], prec)

Use DMRG to calculate the lowest energy eigenstate orthogonal to `orth`
"""
function DMRG(mpo::AbstractMPO, mps_input::LCROpenMPS{T}, orth::Vector{LCROpenMPS{T}} = LCROpenMPS{T}[]; kwargs...) where {T}
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    precision::Float64 = get(kwargs, :precision, DEFAULT_DMRG_precision)
    maxsweeps::Int = get(kwargs, :maxsweeps, 5)

    mps = canonicalize(copy(mps_input))
    set_center!(mps, 1)
    #canonicalize!(mps)
    L = length(mps_input)
    @assert (norm(mps_input) ≈ 1 && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    direction = :right
    Henv = environment(mps, mpo)
    orthenv = [environment(mps, state) for state in orth]
    #Hsquared = mpo*mpo#multiply(mpo, mpo)
    E::real(T), H2::real(T) = real(expectation_value(mps, mpo)), norm(mpo * mps)^2#real(expectation_value(mps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count = 1
    while count <= maxsweeps #TODO make maxcount choosable
        Eprev = E
        mps = sweep(mps, mpo, Henv, orthenv, direction, orth; kwargs...)
        mps = canonicalize(mps, center = center(mps))
        direction = reverse_direction(direction)
        E, H2 = real(expectation_value(mps, mpo)), norm(mpo * mps)^2#real(expectation_value(mps, Hsquared))
        #E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
        if isapprox(E, real(E); atol = precision) && isapprox(H2, real(H2); atol = precision)
            E, H2 = real(E), real(H2)
        else
            @warn "Energies are not real"
        end
        var = H2 - E^2
        println("E, var, ΔE/E = ", E, ", ", var, ", ", (Eprev - E) / E)
        count = count + 1
        if abs((Eprev - E) / E) < precision && var / E^2 < precision #&& count>10
            break
        end
    end

    return mps::LCROpenMPS{T}, E
end


function effective_hamiltonian(mposite, hl, hr, orthvecs)
    szmps = (size(hl, 3), size(mposite, 3), size(hr, 3))
    #Should it be prop to o or v?
    overlap2(o, v) = 100 * o * dot(o, v) #TODO make weight choosable
    function f(v)
        #A = reshape(v, szmps)
        HA = local_mul(hl, hr, mposite, v)
        #overlap(o) = overlap2(o, v)
        # println(size(HA))
        # println(size(v))
        # println(size.(orthvecs))
        OA = mapreduce(o -> overlap2(o, v), naivesum, orthvecs; init = zero(v))

        #OA = sum(overlap, orthvecs; init = zero(v))
        return naivesum(HA, OA)
    end
    return f #LinearMap{complex(eltype(mposite))}(f, prod(szmps), ishermitian = true)
end
naivesum(s1::GenericSite, s2::GenericSite) = GenericSite(data(s1) + data(s2), ispurification(s1))
naivesum(site1::SiteSum, site2::SiteSum) = SiteSum(Tuple([naivesum(s1, s2) for (s1, s2) in zip(sites(site1), sites(site2))]))
naivesum(site1::GenericSite, site2::SiteSum) = SiteSum(Tuple([naivesum(s1, s2) for (s1, s2) in zip(sites(site1), sites(site2))]))
naivesum(site1::SiteSum, site2::GenericSite) = SiteSum(Tuple([naivesum(s1, s2) for (s1, s2) in zip(sites(site1), sites(site2))]))

LinearAlgebra.dot(site1::GenericSite, site2::GenericSite) = dot(data(site1), data(site2))
LinearAlgebra.dot(site1::Union{GenericSite,SiteSum}, site2::Union{GenericSite,SiteSum}) = mapreduce(dot, +, sites(site1), sites(site2))


const BigNumber = Union{ComplexDF64,ComplexDF32,ComplexDF16,Double64,Double32,Double16,BigFloat,Complex{BigFloat}}
# function eigs(heff::LinearMap, x0, nev, prec)
#     if prod(size(heff)) < 100
#         evals, evecs = _eigs_small(Matrix(heff))
#     else
#         evals, evecs = _eigs_large(heff, x0, nev, prec)
#     end
#     T = eltype(heff)
#     return evals::Vector{eltype(heff)}, evecs::Matrix{eltype(heff)}
# end
# function _eigs_small(heff)
#     T = eltype(heff)
#     vals, vecs = eigen(heff)
#     return T.(vals)::Vector{T}, vecs::Matrix{T}
# end
# function _eigs_large(heff::LinearMap, x0, nev, prec)
#     evals::Vector{eltype(heff)}, evecs::Vector{Vector{eltype(heff)}} = eigsolve(heff, vec(x0), nev, :SR, tol = prec, ishermitian = true, maxiter = 3, krylovdim = 20)
#     evecsvec::Matrix{eltype(heff)} = reduce(hcat, evecs)
#     return evals, evecsvec
# end
# function _eigs_large(heff::LinearMap{<:BigNumber}, x0, nev, prec)
#     vals, vecs = partialeigen(partialschur(heff, nev = nev, which = SR(), tol = prec)[1])
#     return vals::Vector{eltype(heff)}, vecs::Matrix{eltype(heff)}
# end
function eigensite(site::S, mposite, hl, hr, orthvecs, prec) where {S<:AbstractSite{<:BigNumber}}
    szmps = size(site)
    heff = effective_hamiltonian(mposite, hl, hr, orthvecs)
    heff2(v) = vec(heff(S(Array(reshape(v, szmps)), ispurification(site))))
    lh = LinearMap{eltype(site)}(heff2, prod(szmps))
    vals, vecs = partialeigen(partialschur(lh, nev = 1, which = SR(), tol = prec)[1])
    vecmin = vecs[:, 1]
    return S(reshape(vecmin, szmps), ispurification(site)) / norm(vecmin), real(vals[1])

    # evals, evecs = eigsolve(heff, site, 1, :SR, tol = prec, ishermitian = true, maxiter = 3, krylovdim = 20)
    # e::eltype(site) = evals[1]
    # vecmin = evecs[1] #::Vector{eltype(hl)}
    # if !(isapprox(e, real(e), atol = prec))
    #     error("ERROR: complex eigenvalues: $e")
    # end
    # return vecmin / norm(vecmin), real(e)
end
function eigensite(site::AbstractSite, mposite, hl, hr, orthvecs, prec)
    #szmps = size(site)
    heff = effective_hamiltonian(mposite, hl, hr, orthvecs)
    evals, evecs = eigsolve(heff, site, 1, :SR, tol = prec, ishermitian = true, maxiter = 3, krylovdim = 20)
    e::eltype(site) = evals[1]
    vecmin = evecs[1] #::Vector{eltype(hl)}
    if !(isapprox(e, real(e), atol = prec))
        error("ERROR: complex eigenvalues: $e")
    end
    return vecmin / norm(vecmin), real(e)
end

""" sweeps from left to right in the DMRG algorithm """
function sweep(mps::LCROpenMPS{T}, mpo::AbstractMPO, Henv::AbstractFiniteEnvironment, orthenv, dir, orth::Vector{LCROpenMPS{T}} = LCROpenMPS{T}[]; kwargs...) where {T}
    L::Int = length(mps)
    shifter = get(kwargs, :shifter, ShiftCenter())
    precision = get(kwargs, :precision, DEFAULT_DMRG_precision)
    if dir == :right
        itr = 1:1:L-1
        # dirval=1
    else
        if dir !== :left
            @error "In sweep: choose dir :left or :right"
        end
        itr = L:-1:2
        # dirval=-1
    end
    for j in itr
        @assert iscenter(mps, j) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        #orthvecs = [vec(data(local_mul(oe.L[j]', oe.R[j]', o[j]))) for (oe, o) in zip(orthenv, orth)]
        # orthvecs = [vec(data(local_mul(orthenv[k].L[j]', orthenv[k].R[j]',orth[k][j]))) for k in 1:N_orth] #FIXME maybe conjugate orthenv from the start?
        # enew = transpose(transfer_matrix(mps[j]', mpo[j], mps[j]) * vec(Henv.R[j])) * vec(Henv.L[j])
        #mps[j], e2 = eigensite(mps[j], mpo[j], Henv.L[j], Henv.R[j], orthvecs, precision)
        orthsites = [local_mul(oe.L[j], oe.R[j], o[j]) for (oe, o) in zip(orthenv, orth)]
        mps[j], e2 = eigensite(mps[j], mpo[j], Henv.L[j], Henv.R[j], orthsites, precision)

        shift_center!(mps, j, dir, shifter; mpo = mpo, env = Henv)
        update! = dir == :right ? update_left_environment! : update_right_environment!
        update!(Henv, j, (mps[j],), (mpo[j], mps[j]))
        # for k in 1:N_orth
        #     update!(orthenv[k],j, orth[k][j], mps[j])
        # end
        for (oe, o) in zip(orthenv, orth)
            update!(oe, j, (mps[j],), (o[j],))
        end
    end
    return mps::LCROpenMPS{T}
end

"""
    eigenstates(hamiltonian, mps, n, prec)

Return the `n` eigenstates and energies with the lowest energy
"""
function eigenstates(hamiltonian::MPO, mps::LCROpenMPS{T}, n::Integer; shifter = ShiftCenter(), kwargs...) where {T}
    #T = eltype(data(mps[1]))
    states = Vector{LCROpenMPS{T}}(undef, n)
    energies = Vector{real(promote_type(T, eltype(hamiltonian[1])))}(undef, n)
    for k = 1:n
        @time state, E = DMRG(hamiltonian, mps, states[1:k-1]; shifter = deepcopy(shifter), kwargs...)
        states[k] = state
        energies[k] = E
    end
    return states, energies
end

function eigenstates2(hamiltonian::MPO, mps::LCROpenMPS{T}, n::Integer; shifter = ShiftCenter(), kwargs...) where {T}
    #T = eltype(data(mps[1]))
    states = Vector{LCROpenMPS{T}}(undef, n)
    energies = Vector{real(promote_type(T, eltype(hamiltonian[1])))}(undef, n)
    for k = 1:n
        @time state, E = DMRG2(hamiltonian, mps, states[1:k-1]; shifter = deepcopy(shifter), kwargs...)
        states[k] = state
        energies[k] = E
    end
    return states, energies
end

const DEFAULT_DMRG_precision = 1e-12

function expansion_term(alpha, site, env, mposite)
    @tensor P[:] := data(site)[1, 2, -3] * env[-1, 4, 1] * data(mposite)[4, -2, 2, -4]
    return alpha * reshape(P, size(env, 1), size(site, 2), size(mposite, 4) * size(site, 3))
end
function expansion_term(alpha, site, env, mposite::ScaledIdentityMPOsite)
    s = size(site)
    newsite = reshape(env * reshape(data(site), s[1], s[2] * s[3]), s[1], s[2], s[3])
    return rmul!(newsite, alpha * data(mposite))
    #@tensor P[:] := data(site)[1,-2,-3]*env[-1,1]
    #return data(mposite)*alpha*reshape(P, size(env,1), size(site,2), size(site,3))
end

function subspace_expand(alpha, site, nextsite, env, mposite, trunc, dir)
    if dir == :left
        site = reverse_direction(site)
        nextsite = reverse_direction(nextsite)
        mposite = reverse_direction(mposite)
    end
    ss = size(site)
    ss2 = size(nextsite)
    d = ss[2]
    P = expansion_term(alpha, site, env, mposite)
    sp = size(P)
    P0 = zeros(eltype(env), sp[3], d, ss2[3])
    M = Array{eltype(env),3}(undef, (ss[1], ss[2], ss[3] + sp[3]))
    B = Array{eltype(env),3}(undef, (ss2[1] + sp[3], ss2[2], ss2[3]))
    for k in 1:d
        M[:, k, :] = hcat(data(site)[:, k, :], P[:, k, :])
        B[:, k, :] = vcat(data(nextsite)[:, k, :], P0[:, k, :])
    end
    U, S, V, err = split_truncate!(reshape(M, ss[1] * ss[2], ss[3] + sp[3]), trunc)
    newsite = GenericSite(reshape(Matrix(U), ss[1], ss[2], length(S)), false)
    newnextsite = VirtualSite(Diagonal(S) * V) * GenericSite(B, false)
    if dir == :left
        newsite = reverse_direction(newsite)
        newnextsite = reverse_direction(newnextsite)
    end
    return newsite, newnextsite
end



function DMRG2(mpo::MPO, mps_input::LCROpenMPS{T}, orth::Vector{LCROpenMPS{T}} = LCROpenMPS{T}[]; kwargs...) where {T}
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    precision::Float64 = get(kwargs, :precision, DEFAULT_DMRG_precision)
    maxsweeps::Int = get(kwargs, :maxsweeps, 5)
    maxbonds::Vector{Int} = get(kwargs, :maxbonds, [mps_input.truncation.Dmax])
    mps = canonicalize(copy(mps_input))
    #canonicalize!(mps)
    set_center!(mps, 1)
    L = length(mps_input)
    @assert (norm(mps_input) ≈ 1 && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    direction = :right
    Henv = environment(mps, mpo)
    orthenv = [environment(state, mps) for state in orth]
    # Hsquared = multiplyMPOs(mpo, mpo)
    E::real(T), H2 = real(expectation_value(mps, mpo)), norm(mpo * mps)^2
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count = 1

    while count <= maxsweeps
        Eprev = E
        if count <= length(maxbonds)
            mps.truncation.Dmax = maxbonds[count]
        end
        mps = twosite_sweep(mps, mpo, Henv, orthenv, direction, orth; kwargs...)
        #mps = canonicalize(mps)
        direction = reverse_direction(direction)
        E, H2 = real(expectation_value(mps, mpo)), norm(mpo * mps)^2
        #E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
        if isapprox(E, real(E); atol = precision) && isapprox(H2, real(H2); atol = precision)
            E, H2 = real(E), real(H2)
        else
            @warn "Energies are not real"
        end
        var = H2 - E^2
        println("Sweep: ", count, ". E, var, ΔE/E = ", E, ", ", var, ", ", (Eprev - E) / E, ". Max bonddim: ", maximum(size.(mps, 1)))
        count = count + 1
        if abs((Eprev - E) / E) < precision && var / E^2 < precision #&& count>10
            break
        end
    end

    return canonicalize(mps)::LCROpenMPS{T}, E
end

""" sweeps from left to right in the DMRG algorithm """
function twosite_sweep(mps::LCROpenMPS{T}, mpo::AbstractMPO, Henv::AbstractFiniteEnvironment, orthenv, dir, orth::Vector{LCROpenMPS{T}} = LCROpenMPS{T}[]; kwargs...) where {T}
    L::Int = length(mps)
    precision = get(kwargs, :precision, DEFAULT_DMRG_precision)
    if dir == :right
        itr = 1:1:L-1
        update! = update_left_environment!
        # dirval=1
    else
        if dir !== :left
            @error "In sweep: choose dir :left or :right"
        end
        itr = L-1:-1:1
        update! = update_right_environment!
        # dirval=-1
    end
    for j in itr
        j1, j2 = dir == :right ? (j, j + 1) : (j + 1, j)
        @assert iscenter(mps, j1) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        orthvecs = [twosite_orthvec(oe.L[j]', oe.R[j+1]', data(o[j]), data(o[j+1])) for (oe, o) in zip(orthenv, orth)]
        A, Λ, B = twosite_eigensite(mps[j], mps[j+1], mpo[j], mpo[j+1], Henv.L[j], Henv.R[j+1], orthvecs, precision, mps.truncation)
        if dir == :right
            shift_center!(mps, 1)
            mps[j] = A
            mps[j+1] = Λ * B
        else
            shift_center!(mps, -1)
            mps[j] = A * Λ
            mps[j+1] = B
        end
        update!(Henv, j1, (mps[j1],), (mpo[j1], mps[j1]))
        # update!(Henv,j2,mps[j2],mpo[j2],mps[j2])
        for (oe, o) in zip(orthenv, orth)
            update!(oe, j1, (o[j1],), (mps[j1],))
            # update!(oe,j2,o[j2],mps[j2])
        end
    end
    return mps::LCROpenMPS{T}
end

function twosite_mpo_application(hl, hr, mpol, mpor, twosite)
    #@tensoropt (lm,dc,rd,-1,-4) out[-1,-2,-3,-4] := hl[-1,lm,ld] * mpol[lm,-2,msl,c] * mpor[c,-3,msr,rm] * sitel[ld, msl, dc]*siter[dc,msr,rd] * hr[-4,rm,rd] 
    # println.(size.([hl,mpol,mpor,twosite,hr]))
    @tensoropt (lm, rd, -1, -4) out[-1, -2, -3, -4] := hl[-1, lm, ld] * mpol[lm, -2, msl, c] * mpor[c, -3, msr, rm] * twosite[ld, msl, msr, rd] * hr[-4, rm, rd]
end
function twosite_orthvec(L, R, sl, sr)
    @tensoropt (lm, dc, rd, -1, -4) out[-1, -2, -3, -4] := L[-1, ld] * sl[ld, -2, dc] * sr[dc, -3, rd] * R[-4, rd]
    return vec(out)
end

function twosite_effective_hamiltonian(mpol, mpor, hl, hr, orthvecs)
    blocksize = (size(hl, 3), size(mpol, 3), size(mpor, 3), size(hr, 3))
    overlap2(o, v) = 100 * o * (o' * v) #TODO make weight choosable
    function f(v)
        A = reshape(v, blocksize)
        HA = twosite_mpo_application(hl, hr, data(mpol), data(mpor), A)
        overlap(o) = overlap2(o, v)
        OA = sum(overlap, orthvecs; init = zero(v))
        return vec(HA) + OA
    end
    return LinearMap{eltype(hl)}(f, prod(blocksize), ishermitian = true)
end
function twosite_eigensite(siteL, siteR, mpoL, mpoR, hL, hR, orthvecs, prec, truncation)
    @tensor block[:] := data(siteL)[-1, -2, 1] * data(siteR)[1, -3, -4]
    blocksize = size(block)
    heff = twosite_effective_hamiltonian(mpoL, mpoR, hL, hR, orthvecs)
    evals, evecs = eigs(heff, block, 1, prec)
    e::eltype(hL) = evals[1]
    vecmin::Vector{eltype(hL)} = evecs[:, 1]
    if !(e ≈ real(e))
        error("ERROR: complex eigenvalues")
    end
    theta = reshape(vecmin, blocksize[1] * blocksize[2], blocksize[3] * blocksize[4])
    U, S, Vt, Dm, err = split_truncate!(theta / norm(theta), truncation)
    Ss = LinkSite(S)
    Us = GenericSite(Array(reshape(U, blocksize[1], blocksize[2], Dm)), ispurification(siteL))
    Vts = GenericSite(Array(reshape(Vt, Dm, blocksize[3], blocksize[4])), ispurification(siteR))

    # @tensor checkcan[:] := inv(data(siteL.Λ1))[-2,5]*inv(data(siteL.Λ1))[-1,1]*block[1,3,2,4]*conj(block[5,3,2,4])
    # println("Block canonical?", norm(checkcan - Matrix(I,blocksize[1],blocksize[1]))) #Yes it is
    # thetaN = theta/norm(theta)
    # @tensor checkcan[:] := inv(data(siteL.Λ1))[-1,1]*(thetaN*thetaN')[1,2] * inv(data(siteL.Λ1))[-2,2]
    # println("Block canonical?", norm(checkcan - Matrix(I,blocksize[1],blocksize[1]))) 
    # println(norm(theta))
    # println(norm((U*Diagonal(S)*Vt)))
    # println("SVD works?: ", norm(theta - (U*Diagonal(S)*Vt))) #No it does not

    # U2 = inv(siteL.Λ1)*Us
    # Vt2 = Vts*inv(siteR.Λ2)
    # Γ1new = OrthogonalLinkSite(siteL.Λ1, U2, Ss)
    # Γ2new = OrthogonalLinkSite(Ss, Vt2, siteR.Λ2)
    # println("R",isrightcanonical(Γ1new))
    # println("L",isleftcanonical(Γ1new))
    return Us, Ss, Vts
end