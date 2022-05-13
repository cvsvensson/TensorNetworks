abstract type AbstractMixer end

struct ShiftCenter <: AbstractMixer end

mutable struct SubspaceExpand <: AbstractMixer #TODO really check if this works as intended
    alpha::Float64
    rate::Float64
    oldmin::Union{Float64,Nothing}
    function SubspaceExpand(alpha, rate)
        @assert rate < 1
        return new(alpha, rate, nothing)
    end
end
SubspaceExpand(alpha) = SubspaceExpand(alpha, 9 / 10)

# copy(s::SubspaceExpand) = SubspaceExpand(copy(s.alpha), copy(s.rate))
# copy(s::ShiftCenter) = ShiftCenter()

function shift_center!(mps, j, dir, ::ShiftCenter; kwargs...)
    if dir == :right
        shift_center_right!(mps)
    elseif dir == :left
        shift_center_left!(mps)
    end
end

function shift_center!(mps, j, dir, SE::SubspaceExpand; mpo, env, kwargs...)
    newmin = transpose(_local_transfer_matrix((mps[j],), (mpo[j], mps[j]),:left) * vec(env.R[j])) * vec(env.L[j])
    if SE.alpha < mps.truncation.tol
        shift_center!(mps, j, dir, ShiftCenter(); mpo, env)
        return SE.oldmin
    end
    if dir == :right
        dirval = +1
        j1 = j
        j2 = j + 1
    elseif dir == :left
        dirval = -1
        j1 = j - 1
        j2 = j
    end
    A, B = subspace_expand(SE.alpha, mps[j], mps[j+dirval], env[j, reverse_direction(dir)], mpo[j], mps.truncation, dir)
    mps.center += dirval
    mps.Γ[j] = A
    mps.Γ[j+dirval] = B
    T = prod(transfer_matrices((mps[j1:j2],), (mpo[j1:j2], mps[j1:j2]), :left))
    truncmin = transpose(T * vec(env.R[j2])) * vec(env.L[j1])

    if SE.oldmin !== nothing
        if real((truncmin - newmin) / (SE.oldmin - newmin)) > 0.3
            SE.alpha *= SE.rate
        else
            SE.alpha *= 1 / (SE.rate)^(1 / 10)
        end
    end
    # println(SE.alpha)
    SE.oldmin = real.(truncmin)
end

function iterative_compression(target::AbstractMPS, guess::AbstractMPS, prec = 1e-8; maxiter = 50, shifter = ShiftCenter)
    env = environment(guess, target)
    # mps = guess
    #    mps = iscanonical(guess) ? guess :  canonicalize(guess)
    mps = canonicalize(guess)
    set_center!(mps, 1)
    dir = :right
    #targetnorm = norm(target)
    # IL(site) = Array(vec(Diagonal{eltype(guess[1])}(I,size(site,1))))
    # IR(site) = Array(vec(Diagonal{eltype(guess[1])}(I,size(site,3))))
    errorfunc(mps) = 1 - abs(scalar_product(target, mps)) #FIXME can save memory by using precomputed envuronments

    #TODO Make it work for UMPS. The following errorfunction can be used
    #density_matrix(mps,k) = @tensor rho[:] := data(mps[k])[1,-1,2]*conj(data(mps[k])[1,-2,2])
    #errorfunc(mps) = real(1- sum([tr(density_matrix(target,k)*density_matrix(mps,k)) for k in 1:length(mps)]))

    #real(targetnorm - sum([transpose(transfer_matrix(site) * IR(site))*IL(site) for site in mps[1:end]])) #use @view
    #if isinfinite(target)
    count = 1
    error = errorfunc(mps)
    # error = error 
    println(error)
    while error > prec && count < maxiter
        mps, env = sweep(target, mps, env, dir, prec)
        newerror = errorfunc(mps)
        if abs(error - newerror) < prec
            break
        end
        error = newerror
        println(error)
        dir = reverse_direction(dir)
    end
    return mps
end

function sweep(target, mps, env, dir, prec; kwargs...)
    L = length(mps)
    shifter = get(kwargs, :shifter, ShiftCenter())
    if dir == :right
        itr = 1:L-1+isinfinite(mps)
        dirval = 1
    elseif dir == :left
        itr = L:-1:2-isinfinite(mps)
        dirval = -1
    else
        @error "In sweep: choose dir :left or :right"
    end
    for j in itr
        @assert (iscenter(mps, j)) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        newsite = local_mul(env.L[j], env.R[j], target[j])
        mps[j] = newsite / norm(newsite)
        shift_center!(mps, j, dir, shifter; error = error)
        update! = dir == :right ? update_left_environment! : update_right_environment!
        update!(env, j, (mps[j],), (target[j],))
    end
    return mps, env
end
