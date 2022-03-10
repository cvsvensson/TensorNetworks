
transfer_matrix_bond(mps::AbstractVector{<:OrthogonalLinkSite}, site::Integer, dir::Symbol) = data(link(mps[site], :left))
transfer_matrix_bond(mps::AbstractVector{<:GenericSite}, site::Integer, dir::Symbol) = I#Diagonal(I,size(mps[site],1))
transfer_matrix_bond(mps1::AbstractMPS, mps2::AbstractMPS, site::Integer, dir::Symbol) = kron(transfer_matrix_bond(mps1, site, dir), transfer_matrix_bond(mps2, site, dir))
transfer_matrix_bond(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, site::Integer, dir::Symbol) = kron(transfer_matrix_bond(mps1, site, dir), transfer_matrix_bond(mpo, site, dir), transfer_matrix_bond(mps2, site, dir))

Base.kron(a::UniformScaling, b::UniformScaling) = a * b
Base.kron(a::UniformScaling, b::AbstractMatrix) = Diagonal(a, size(b, 1)) * b
Base.kron(a::AbstractMatrix, b::UniformScaling) = Diagonal(b, size(a, 1)) * a

transfer_matrix_bond(site::OrthogonalLinkSite, dir::Symbol) = data(link(site, :left))
transfer_matrix_bond_dense(site::OrthogonalLinkSite, dir::Symbol) = transfer_matrix_bond(site,dir)
transfer_matrix_bond_dense(site::GenericSite{T}, dir::Symbol) where {T} = Matrix(one(T)I,size(site,1))

# %% Transfer Matrices
"""
	transfer_left(Γ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""
# function transfer_left2(Γ::Array{T,3}) where {T}
#     dims = size(Γ)
#     function func(Rvec)
#         Rtens = reshape(Rvec,dims[3],dims[3])
#         @tensoropt (t1,b1,-1,-2) temp[:] := Rtens[t1,b1]*conj(Γ[-1, c1, t1])*Γ[-2, c1, b1]
#         return vec(temp)
#     end
# 	function func_adjoint(Lvec)
# 		Ltens = reshape(Lvec,dims[1],dims[1])
# 		@tensoropt (t1,b1,-1,-2) temp[:] := Ltens[t1,b1]*Γ[t1, c1, -1]*conj(Γ[b1, c1, -2])
# 		return vec(temp)
# 	end
#     return LinearMap{T}(func,func_adjoint, dims[1]^2,dims[3]^2)
# end
# _transfer_left_mpo(Γ1::MPOsite, Γ2::MPOsite) = _transfer_left_mpo(data(Γ1), data(Γ2))
_transfer_left_mpo(s1::OrthogonalLinkSite, op::MPOsite, s2::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s1, :right), op, GenericSite(s2, :right))
_transfer_left_mpo(s1::OrthogonalLinkSite, op::MPOsite) = _transfer_left_mpo(GenericSite(s1, :right), op, GenericSite(s1, :right))
_transfer_left_mpo(s1::OrthogonalLinkSite, s2::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s1, :right), GenericSite(s2, :right))
_transfer_left_mpo(s::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s, :right))
function __transfer_left_mpo(Γ1::GenericSite, Γ2::GenericSite)
    dims1 = size(Γ1)
    dims2 = size(Γ2)
    function func(R::BoundaryVector{<:Number,3})
        #Rtens = reshape(Rvec, dims1[3], dims2[3])
        @tensoropt (t1, b1, -1, -2) temp[:] := data(R)[t1, b1] * conj(data(Γ1)[-1, c1, t1]) * data(Γ2)[-2, c1, b1]
        return BoundaryVector(temp)
    end
    function func_adjoint(L::BoundaryVector{<:Number,3})
        #Ltens = reshape(Lvec, dims1[1], dims2[1])
        @tensoropt (t1, b1, -1, -2) temp[:] := data(L)[t1, b1] * data(Γ1)[t1, c1, -1] * conj(data(Γ2)[b1, c1, -2])
        return BoundaryVector(temp)
    end
    return LinearMap{eltype(Γ1)}(func, func_adjoint, dims1[1] * dims2[1], dims1[3] * dims2[3])
end
function __transfer_left_mpo(Γ1::GenericSite)
    dims = size(Γ1)
    Γ = data(Γ1)
    function func(R::BoundaryVector{<:Number,2})
        #Rtens = reshape(Rvec, dims[3], dims[3])
        @tensoropt (t1, b1, -1, -2) temp[:] := data(R)[t1, b1] * conj(Γ[-1, c1, t1]) * Γ[-2, c1, b1]
        return BoundaryVector(temp)
    end
    function func_adjoint(L::BoundaryVector{<:Number,2})
        #Ltens = reshape(Lvec, dims[1], dims[1])
        @tensoropt (t1, b1, -1, -2) temp[:] := data(L)[t1, b1] * Γ[t1, c1, -1] * conj(Γ[b1, c1, -2])
        return BoundaryVector(temp)
    end
    return LinearMap{eltype(Γ1)}(func, func_adjoint, dims[1]^2, dims[3]^2)
end

# function __transfer_left_mpo(Rvec,g1,g2,s1,s2)
# 	Rtens = reshape(Rvec,s1[3],s2[3])
# 	R2 = reshape(g1*Rtens, s1[1], s1[2]*s1[3])
# 	return vec(R2*g2)
# end
# function __transfer_left_adjoint_mpo(Lvec,g1l,g2l,s1,s2)
# 	Ltens = transpose(reshape(Lvec,s1[1],s2[1]))
# 	L2 = reshape(Ltens*g1l, s2[1]*s1[2],s1[3])
# 	return vec(transpose(g2l*L2))
# end
# function _transfer_left_mpo(Γ1::GenericSite, Γ2::GenericSite)
#     s1 = size(Γ1)
# 	s2 = size(Γ2)
# 	g1 = reshape(data(Γ1), s1[1]*s1[2],s1[3])
# 	g2 = reshape(permutedims(data(Γ2),[2,3,1]),s2[2]*s2[3],s2[1])

# 	g1l = reshape(conj(data(Γ1)), s1[1],s1[2]*s1[3])
# 	g2l = reshape(permutedims(conj(data(Γ2)),[3,1,2]),s2[3],s2[2]*s2[1])
# 	func(R) = __transfer_left_mpo(R,g1,g2,s1,s2)
# 	func_adjoint(L) = __transfer_left_adjoint_mpo(L,g1l,g2l,s1,s2)
#     return LinearMap{eltype(Γ1)}(func,func_adjoint, s1[1]*s2[1],s1[3]*s2[3])
# end
# function _transfer_left_mpo(Γ1::GenericSite)
#     s1 = size(Γ1)
# 	s2 = size(Γ1)
# 	g1 = reshape(data(Γ1), s1[1]*s1[2],s1[3])
# 	g2 = reshape(permutedims(data(Γ1),[2,3,1]),s2[2]*s2[3],s2[1])

# 	g1l = reshape(conj(data(Γ1)), s1[1],s1[2]*s1[3])
# 	g2l = reshape(permutedims(conj(data(Γ1)),[3,1,2]),s2[3],s2[2]*s2[1])
# 	func(R) = __transfer_left_mpo(R,g1,g2,s1,s2)
# 	func_adjoint(L) = __transfer_left_adjoint_mpo(L,g1l,g2l,s1,s2)
#     return LinearMap{eltype(Γ1)}(func,func_adjoint, s1[1]*s2[1],s1[3]*s2[3])
# end
# _transfer_left_mpo(Γ1::GenericSite, mpo::MPOsite) = _transfer_left_mpo(Γ1, mpo, Γ1)
_transfer_left_mpo(Γ1, mpo::ScaledIdentityMPOsite, Γ2) = data(mpo) * _transfer_left_mpo(Γ1, Γ2)
_transfer_left_mpo(Γ1, mpo::ScaledIdentityMPOsite) = data(mpo) * _transfer_left_mpo(Γ1)
function __transfer_left_mpo(Γ1::GenericSite, mpo::MPOsite, Γ2::GenericSite)
    dims1 = size(Γ1)
    dims2 = size(Γ2)
    smpo = size(mpo)
    T = promote_type(eltype(Γ1), eltype(mpo), eltype(Γ2))
    function func(R::BoundaryVector{<:Number,3})
        #Rtens = reshape(Rvec, dims1[3], smpo[4], dims2[3])
        @tensoropt (tr, br, -1, -2, -3) temp[:] := conj(data(Γ1)[-1, u, tr]) * data(mpo)[-2, u, d, cr] * data(Γ2)[-3, d, br] * data(R)[tr, cr, br]
        return BoundaryVector(temp)
    end
    function func_adjoint(L::BoundaryVector{<:Number,3})
        #Ltens = reshape(Lvec, dims1[1], smpo[1], dims2[1])
        @tensoropt (bl, tl, -1, -2, -3) temp[:] := data(L)[tl, cl, bl] * data(Γ1)[tl, u, -1] * conj(data(mpo)[cl, u, d, -2]) * conj(data(Γ2)[bl, d, -3])
        return BoundaryVector(temp)
    end
    return LinearMap{T}(func, func_adjoint, smpo[1] * dims1[1] * dims2[1], smpo[4] * dims1[3] * dims2[3])
end


#TODO Check performance vs ncon, or 'concatenated' versions. Ncon is slower. concatenated is faster
function __transfer_left_mpo(mposites::Vararg{MPOsite,N}) where {N}
    #sizes = size.(mposites)
    rs = size.(mposites, 4)
    ls = size.(mposites, 1)
    #ds = size.(mposites,3)
    us = size.(mposites, 2)
    function contract(R::BoundaryVector{<:Number,N})
        site = data(mposites[1])
        # temp = reshape(R, rs[1], prod(rs[2:N])) #0.014480 seconds (250 allocations: 3.682 MiB)
        # @tensor temp[newdone, remaining, down, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

        # temp = reshape(R, rs[1], prod(rs[2:N])) 0.013839 seconds (160 allocations: 3.674 MiB)
        # @tensor temp[down, remaining, newdone, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

        temp = reshape(data(R), rs[1], prod(rs[2:N]))
        @tensor temp[down, remaining, hat, newdone] := site[newdone, hat, down, rc] * temp[rc, remaining]
        for k in 2:N
            site = data(mposites[k])

            temp = reshape(temp, us[k], rs[k], prod(rs[k+1:N]), us[1], prod(ls[1:k-1]))
            @tensor temp[down, remaining, hat, done, newdone] := site[newdone, upc, down, rc] * temp[upc, rc, remaining, hat, done] order = (upc, rc)

            # temp = reshape(temp, us[k], rs[k], prod(rs[k+1:N]), prod(ls[1:k-1]), us[1])  0.013839 seconds (160 allocations: 3.674 MiB)
            # @tensor temp[down, remaining, done,newdone,hat] := site[newdone,upc,down,rc] * temp[upc, rc,remaining,done,hat] order=(upc,rc)

            # temp = reshape(temp, prod(ls[1:k-1]), rs[k], prod(rs[k+1:N]), us[k], us[1])  #0.014480 seconds (250 allocations: 3.682 MiB)
            # @tensor temp[done, newdone, remaining, down, hat] := site[newdone,upc,down,rc] * temp[done,rc,remaining,upc,hat] order=(rc,upc)
        end
        if us[1] != 1
            @tensor temp[:] := temp[1, -1, 1, -2, -3]
        end
        return reshape(temp, prod(ls))
    end
    function adjoint_contract(R::BoundaryVector{<:Number,N})
        temp = reshape(data(R), ls[1], prod(ls[2:N]))
        site = permutedims(conj(mposites[1]), [4, 2, 3, 1])
        @tensor temp[down, remaining, hat, newdone] := site[newdone, hat, down, rc] * temp[rc, remaining]
        for k in 2:N
            site = permutedims(conj(mposites[k]), [4, 2, 3, 1])

            temp = reshape(temp, us[k], ls[k], prod(ls[k+1:N]), us[1], prod(rs[1:k-1]))
            @tensor temp[down, remaining, hat, done, newdone] := site[newdone, upc, down, rc] * temp[upc, rc, remaining, hat, done] order = (upc, rc)
        end
        if us[1] != 1
            @tensor temp[:] := temp[1, -1, 1, -2, -3]
        end
        return reshape(temp, prod(rs))
    end
    map = LinearMap{promote_type(eltype.(mposites)...)}(contract, adjoint_contract, prod(ls), prod(rs))
    return map
end

function _transfer_left_mpo_ncon(mposites::Vararg{MPOsite,N}) where {N}
    rs = size.(mposites, 4)
    ls = size.(mposites, 1)
    us = size.(mposites, 2)
    function contract(R)
        # index(1) = [-1,last,3,1]
        # index(2) = [-2,3,5,2]
        # index(3) = [-3,5,7,4] # or if last: [-3,5,6,4]
        # index(4) = [-4, 7, 9, 6]
        # index(N) = [-N], , , 
        function index(k)
            if N == 1
                return [-1, 2, 2, 1]
            elseif k == 1
                return [-1, 2 * N, 3, 1]
            elseif k == N
                return [-k, 2 * k - 1, 2 * k, 2k - 2]
            else
                return [-k, 2 * k - 1, 2 * k + 1, 2k - 2]
            end
        end
        indexR = [1, [2k - 2 for k in 2:N]...]
        tens = reshape(R, rs)
        ncon([tens, data.(mposites)...], [indexR, [index(k) for k in 1:N]...])
        return reshape(tens, prod(ls))
    end
    function adjoint_contract(R)
        function index(k)
            if N == 1
                return reverse([-1, 2, 2, 1])
            elseif k == 1
                return reverse([-1, 2 * N, 3, 1])
            elseif k == N
                return reverse([-k, 2 * k - 1, 2 * k, 2k - 2])
            else
                return reverse([-k, 2 * k - 1, 2 * k + 1, 2k - 2])
            end
        end
        indexR = [1, [2k - 2 for k in 2:N]...]
        tens = reshape(R, ls)
        ncon([tens, conj.(data.(mposites))...], [indexR, [index(k) for k in 1:N]...])
        return reshape(tens, prod(rs))
    end
    map = LinearMap{promote_type(eltype.(mposites)...)}(contract, adjoint_contract, prod(ls), prod(rs))
    return map
end

_transfer_right_mpo(sites::Vararg{Union{AbstractMPOsite,AbstractSite},N}) where {N} = _transfer_left_mpo(reverse_direction.(sites)...)
reverse_direction(Γ::Array{<:Number,3}) = permutedims(Γ, [3, 2, 1])
reverse_direction(Γs::AbstractVector{<:Union{AbstractMPOsite,AbstractSite}}) = reverse(reverse_direction.(Γs))
function _transfer_left_gate(Γ1, gate::AbstractSquareGate, Γ2)
    # oplength = operatorlength(gate)
    # Γnew1 = copy(reverse([Γ1...]))
    # Γnew2 = copy(reverse([Γ2...]))
    # for k = 1:oplength
    # 	Γnew1[oplength+1-k] = reverse_direction(Γnew1[oplength+1-k])
    # 	Γnew2[oplength+1-k] = reverse_direction(Γnew2[oplength+1-k])
    # 	gate = permutedims(gate,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
    # end
    Γnew1 = reverse_direction(Γ1)
    Γnew2 = reverse_direction(Γ2)
    return _transfer_right_gate(Γnew1, reverse_direction(gate), Γnew2)
end

function _transfer_left_gate(Γ, gate::AbstractSquareGate)
    # oplength = operatorlength(gate)
    # Γnew = copy(reverse([Γ...]))
    Γnew = reverse_direction(Γ)
    # for k = 1:oplength
    # 	# Γnew[oplength+1-k] = reverse_direction(Γnew[oplength+1-k])
    # 	gate = permutedims(gate,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
    # end
    return _transfer_right_gate(Γnew, reverse_direction(gate))
end
_transfer_right_gate(Γ1::AbstractVector{<:OrthogonalLinkSite}, gate::GenericSquareGate) = _transfer_right_gate([GenericSite(Γ, :left) for Γ in Γ1], gate)
_transfer_right_gate(Γ1::AbstractVector{<:OrthogonalLinkSite}, gate::GenericSquareGate, Γ2::AbstractVector{<:OrthogonalLinkSite}) = _transfer_right_gate([GenericSite(Γ, :left) for Γ in Γ1], gate, [GenericSite(Γ, :left) for Γ in Γ2])
function _transfer_right_gate(Γ1::AbstractVector{GenericSite{T}}, gate::GenericSquareGate, Γ2::AbstractVector{GenericSite{T}}) where {T}
    op = data(gate)
    oplength = operatorlength(gate)
    @assert length(Γ1) == length(Γ2) == oplength "Error in transfer_right_gate: number of sites does not match gate length"
    @assert size(gate, 1) == size(Γ1[1], 2) == size(Γ2[1], 2) "Error in transfer_right_gate: physical dimension of gate and site do not match"
    perm = [Int(floor((k + 1) / 2)) + oplength * iseven(k) for k in 1:2*oplength]
    opvec = vec(permutedims(op, perm))
    s_start1 = size(Γ1[1])[1]
    s_start2 = size(Γ2[1])[1]
    s_final1 = size(Γ1[oplength])[3]
    s_final2 = size(Γ2[oplength])[3]
    function T_on_vec(invec::BoundaryVector{<:Number,2}) #FIXME Compare performance to a version where the gate is applied between the top and bottom layer of sites
        v = reshape(invec, 1, s_start1, s_start2)
        for k in 1:oplength
            @tensoropt (1, 2) v[:] := conj(data(Γ1[k]))[1, -2, -4] * v[-1, 1, 2] * data(Γ2[k])[2, -3, -5]
            sv = size(v)
            v = reshape(v, prod(sv[1:3]), sv[4], sv[5])
        end
        #return transpose(opvec)*reshape(v,size(v,1),size(v,2)*size(v,3))
        @tensor vout[:] := v[1, -1, -2] * opvec[1]
        # @tullio vout[a,b] := v[c,a,b] * opvec[c]
        return BoundaryVector(vout)
    end
    #TODO Define adjoint
    return LinearMap{T}(T_on_vec, s_final1 * s_final2, s_start1 * s_start2)
end
function _transfer_right_gate(Γ::AbstractVector{GenericSite{T}}, gate::GenericSquareGate) where {T}
    op = data(gate)
    oplength = operatorlength(gate)
    @assert length(Γ) == oplength "Error in transfer_right_gate: number of sites does not match gate length"
    @assert size(gate, 1) == size(Γ[1], 2) "Error in transfer_right_gate: physical dimension of gate and site do not match"
    perm = [Int(floor((k + 1) / 2)) + oplength * iseven(k) for k in 1:2*oplength]
    opvec = vec(permutedims(op, perm))
    s_start = size(Γ[1])[1]
    s_final = size(Γ[oplength])[3]
    function T_on_vec(invec::BoundaryVector{<:Number,2})
        v = reshape(invec, 1, s_start, s_start)
        for k in 1:oplength
            @tensoropt (1, 2) v[:] := conj(data(Γ[k])[1, -2, -4]) * v[-1, 1, 2] * data(Γ[k])[2, -3, -5]
            sv = size(v)
            v = reshape(v, prod(sv[1:3]), sv[4], sv[5])
        end
        @tensor v[:] := v[1, -1, -2] * opvec[1]
        return BoundaryVector(v)
    end
    #TODO Define adjoint
    return LinearMap{T}(T_on_vec, s_final^2, s_start^2)
end

#Sites 
transfer_matrix(site::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix(tuple(site), dir)
transfer_matrix(site1::AbstractSite, site2::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix((site1, site2), dir)
transfer_matrix(site::AbstractSite, op::AbstractMPOsite, dir::Symbol = :left) = _local_transfer_matrix((site, op), dir)
transfer_matrix(site1::AbstractSite, op::AbstractMPOsite, site2::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix((site1, op, site2), dir)
transfer_matrix(site::AbstractSite, op::ScaledIdentityGate, dir::Symbol = :left) = data(op) * transfer_matrix(site, dir)
transfer_matrix(site1::AbstractSite, op::ScaledIdentityGate, site2::AbstractSite, dir::Symbol = :left) = data(op) * _local_transfer_matrix((site1, site2), dir)

_purify_site(site::AbstractMPOsite, purify::Bool) = purify ? auxillerate(site) : site
_purify_site(site, purify::Bool) = site
function _local_transfer_matrix(sites::Tuple, direction::Symbol)
    # K = promote_type(eltype.(sites)...)
    # purify::Bool = any(([ispurification(site) for site in sites if site isa AbstractSite]))
    # newsites::NTuple{<:Any, <:} = Tuple([_purify_site(site,purify) for site in sites if !(site isa ScaledIdentityMPOsite)])
    # scaling::K = prod([K(data(site)) for site in sites if site isa ScaledIdentityMPOsite], init=one(K))
    # return (scaling*__local_transfer_matrix(newsites,direction))::LinearMap{K}
    purify::Bool = any(([ispurification(site) for site in sites if site isa AbstractSite]))
    newsites = _purify_site.(sites, purify)
    return __local_transfer_matrix(newsites, direction)
end
function __local_transfer_matrix(sites::Tuple, direction::Symbol = :left)
    if direction == :left
        return _transfer_left_mpo(sites...)
    else
        if direction !== :right
            @warn "Defaulting direction to :right"
        end
        return _transfer_right_mpo(sites...)
    end
end

_local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::ScaledIdentityGate, direction::Symbol = :left) = data(op) * prod(transfer_matrices(site1, direction))
_local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::ScaledIdentityGate, site2::AbstractVector{<:AbstractSite}, direction::Symbol = :left) = data(op) * prod(transfer_matrices(site1, site2, direction))
function _local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::AbstractSquareGate, site2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(site1) == length(site2) == operatorlength(op)
    if ispurification(site1[1])
        @assert ispurification(site2[1])
        op = auxillerate(op)
    end
    if direction == :left
        T = _transfer_left_gate(site1, op, site2)
    elseif direction == :right
        T = _transfer_right_gate(site1, op, site2)
    else
        error("Choose direction :left or :right")
    end
    return T
end
function _local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::AbstractSquareGate, direction::Symbol = :left)
    @assert length(site1) == operatorlength(op)
    if ispurification(site1[1])
        op = auxillerate(op)
    end
    if direction == :left
        T = _transfer_left_gate(site1, op)
    elseif direction == :right
        T = _transfer_right_gate(site1, op)
    else
        error("Choose direction :left or :right")
    end
    return T
end

transfer_matrix(site::AbstractSite, op::Union{AbstractSquareGate{<:Any,2},Matrix}; direction::Symbol = :left) = transfer_matrix((site, MPOsite(op)), direction)

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractSquareGate, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2)
    n = operatorlength(op)
    return [_local_transfer_matrix(sites1[k:k+n-1], op, sites2[k:k+n-1], direction) for k in 1:length(sites1)+1-n]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractSquareGate, direction::Symbol = :left)
    n = operatorlength(op)
    return [_local_transfer_matrix(sites1[k:k+n-1], op, direction) for k in 1:length(sites1)+1-n]
end

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractMPOsite, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2)
    return [_local_transfer_matrix((sites1[k], op, sites2[k]), direction) for k in 1:length(sites1)]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractMPOsite, direction::Symbol = :left)
    return [_local_transfer_matrix((sites1[k], op, direction), direction) for k in 1:length(sites1)]
end

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, ops::AbstractVector{<:AbstractSquareGate}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2) == length(ops)
    N = length(sites1)
    ns = operatorlength.(ops)
    return [_local_transfer_matrix(sites1[k:k+ns[k]-1], op, sites2[k:k+ns[k]-1], direction) for (k, op) in enumerate(ops) if !(k + ns[k] - 1 > N)]
    # Ts = LinearMap{numtype(sites1,sites2)}[]
    # for k in 1:N
    # 	n = length(op[k])
    # 	if k+n-1>N
    # 		break
    # 	end
    # 	Tm = _local_transfer_matrix(view(sites1,k:k+n-1), op[k], view(sites2,k:k+n-1), direction)
    # 	push!(Ts,Tm)
    # end
    # Ts
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, ops::AbstractVector{<:AbstractSquareGate}, direction::Symbol = :left)
    @assert length(sites1) == length(ops)
    N = length(sites1)
    # Ts = LinearMap{numtype(sites1)}[]
    ns = operatorlength.(ops)
    return [_local_transfer_matrix(sites1[k:k+ns[k]-1], op, direction) for (k, op) in enumerate(ops) if !(k + ns[k] - 1 > N)]
    # for k in 1:N
    # 	n = length(op[k])
    # 	if k+n-1>N
    # 		break
    # 	end
    # 	Tm = _local_transfer_matrix(view(sites1,k:k+n-1), op[k], direction)
    # 	push!(Ts,Tm)
    # end
    # Ts
end

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::Union{AbstractMPO,AbstractVector{<:MPOsite}}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2) == length(op)
    return [_local_transfer_matrix(sos, direction) for sos in zip(sites1, op, sites2)]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::Union{AbstractMPO,AbstractVector{<:MPOsite}}, direction::Symbol = :left)
    @assert length(sites1) == length(op)
    return [_local_transfer_matrix(so, direction) for so in zip(sites1, op)]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2)
    return [_local_transfer_matrix(ss, direction) for ss in zip(sites1, sites2)]
end
transfer_matrices(sites::AbstractVector{<:AbstractSite}, direction::Symbol = :left) = [_local_transfer_matrix(tuple(site), direction) for site in sites]


function transfer_matrix(sites1::AbstractVector{<:AbstractSite}, op, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    Ts = transfer_matrices(sites1, op, sites2, direction)
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return prod(Ts) #Products of many linear operators cause long compile times!
end
function transfer_matrix(sites1::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    Ts = transfer_matrices(sites1, direction)
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return prod(Ts) #Products of many linear operators cause long compile times!
end
function transfer_matrix(sites1::AbstractVector{<:AbstractSite}, op, direction::Symbol = :left)
    Ts = transfer_matrices(sites1, op, direction)
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return prod(Ts) #Products of many linear operators cause long compile times!
end

# transfer_matrix(sites1::AbstractVector{<:AbstractSite}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrix(sites1, IdentityMPO(length(sites1)), sites2, direction)
# transfer_matrix(sites::AbstractVector{<:AbstractSite},op, direction::Symbol=:left) = transfer_matrix(sites,op,sites,direction)
# transfer_matrix(sites::AbstractVector{<:AbstractSite}, direction::Symbol=:left) = transfer_matrix(sites,IdentityMPO(length(sites1)),direction)


# This function gives the transfer matrix for a single site which acts on the right.
# """
# 	transfer_matrix_squared(A)

# Return the transfer matrix for the tensor `A` squared
# """
# function transfer_matrix_squared(A)
#     sA=size(A)
#     function contract(R)
#         temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
#         @tensoropt (r,-2,-3,-4) begin
#             temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
#             temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
#             temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
#             temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
#             temp[:] := temp[-1,-2,-3,-4,c,c]
#         end
#         st = size(temp)
#         return reshape(temp,st[1]*st[2]*st[3]*st[4])
#     end
#     T = LinearMap{ComplexF64}(contract,sA[1]^4,sA[4]^4)
#     return T
# end


# """ #FIXME replace A by gamma lambda
# 	transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right)

# Return the transfer matrix for the density matrix squared
# """
# function transfer_matrix_squared(Γ::Array{T,3}, Λ::Array{T,1}, dir=:right) where {T}
# 	sA=size(Γ)
# 	d=Int(sqrt(sA[2]))
# 	A = reshape(A,sA[1],d,d,sA[3])
# 	if dir==:right
# 		A = permutedims(A,[4,2,3,1])
# 	end
# 	A = reshape(Λ,1,1,dims[3]) .* A
#     function contract(R)
#         temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
#         @tensoropt (r,-2,-3,-4) begin
#             temp[:] := temp[r,-2,-3,-4]*A[r,-6,-5,-1]
#             temp[:] := temp[-1,r,-3,-4,c,-6]*conj(A[r,-5,c,-2])
#             temp[:] := temp[-1,-2,r,-4,c,-6]*A[r,c,-5,-3]
#             temp[:] := temp[-1,-2,-3,r,c,-6]*conj(A[r,-5,c,-4])
#             temp[:] := temp[-1,-2,-3,-4,c,c]
#         end
#         st = size(temp)
#         return reshape(temp,st[1]*st[2]*st[3]*st[4])
#     end
#     return LinearMap{T}(contract,sA[1]^4,sA[4]^4)
# end
