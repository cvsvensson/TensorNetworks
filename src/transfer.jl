#TransferMatrix(f::Function) = TransferMatrix{typeof(f)}(f)
TransferMatrix(f::Union{Function,Nothing}, fa::Union{Function,Nothing}, T::DataType, s) = TransferMatrix{typeof(f),typeof(fa),T,typeof(s)}(f, fa, s)

Base.eltype(::AbstractTransferMatrix{T}) where {T} = T
Base.size(T::AbstractTransferMatrix) = T.sizes
Base.size(T::AbstractTransferMatrix, i) = T.sizes[i]
Base.:*(T::TransferMatrix, v) = T.f(v)
Base.:*(T::CompositeTransferMatrix, v) = foldr(*, T.maps, init = v)
Base.adjoint(T::TransferMatrix) = TransferMatrix(T.fa, T.f, eltype(T), (size(T, 2), size(T, 1)))
Base.adjoint(T::CompositeTransferMatrix{<:Any,K,S}) where {K,S} = CompositeTransferMatrix{K,S}(reverse(adjoint.(T.maps)))

# Base.:*(x::Number, T::TransferMatrix) = TransferMatrix(y -> x * T.f(y), eltype(T), size(T))
Base.:*(x::K, T::AbstractTransferMatrix) where {K<:Number} = TransferMatrix(v -> x * v, v -> conj(x) * v, K, (size(T, 1), size(T, 1))) * T
# Base.:*(x::Number, T::TransferMatrixAdj) = TransferMatrix(y -> x * T.f(y), y -> conj(x) * T.fa(y), eltype(T), size(T))

# Base.:*(T1::TransferMatrix, T2::TransferMatrix) = TransferMatrix(T1.f ∘ T2.f, promote_type(eltype.((T1, T2))...), (size(T1, 1), size(T2, 2)))

Base.:*(T1::TransferMatrix, T2::TransferMatrix) = CompositeTransferMatrix{promote_type(eltype(T1), eltype(T2))}(tuple(T1, T2))
Base.:*(T1::TransferMatrix, T2::CompositeTransferMatrix) = CompositeTransferMatrix{promote_type(eltype(T1), eltype(T2))}(tuple(T1, T2.maps...))
Base.:*(T1::CompositeTransferMatrix, T2::TransferMatrix) = CompositeTransferMatrix{promote_type(eltype(T1), eltype(T2))}(tuple(T1.maps..., T2))

# Base.:*(T1::TransferMatrix, T2::TransferMatrix) = TransferMatrix(T1.f ∘ T2.f, promote_type(eltype.((T1, T2))...), (size(T1, 1), size(T2, 2)))
#Base.:*(T1::TransferMatrixAdj{F1,Fa1,K1,S}, T2::TransferMatrixAdj{F2,Fa2,K2,S}) where {F1,F2,Fa1,Fa2,K1,K2,S} = 
# TransferMatrixAdj{ComposedFunction{F1,F2},ComposedFunction{Fa2,Fa1},promote_type(K1, K2),S}(T1.f ∘ T2.f, T2.fa ∘ T1.fa, (size(T1, 1), size(T2, 2)))

IdentityTransferMatrix(T, s) = TransferMatrix(identity, identity, T, s)

IdentityBoundary(T, sizes::Array{NTuple{2,Int},2}) = BlockBoundaryVector{T,2}([IdentityBoundary(T, size) for size in sizes])
IdentityBoundary(T, size::NTuple{2,Int}) = Matrix(one(T)I, size...)

function Matrix(T::AbstractTransferMatrix{Num,NTuple{2,Array{NTuple{N,Int},K}}}) where {Num,N,K}
    s = size(T)
    idim = sum(prod.(s[2]))
    odim = sum(prod.(s[1]))
    v = zeros(Num, idim)
    v[1] = 1
    m = zeros(Num, odim, idim)
    for k in 1:idim
        m[:, k] = vec(T * BlockBoundaryVector(v, s[2]))
        v = circshift(v, 1)
    end
    return m
end
function Matrix(T::AbstractTransferMatrix{Num,NTuple{2,NTuple{N,Int}}}) where {Num,N}
    odims, idims = size(T)
    v = vec(zeros(Num, idims))
    v[1] = 1
    m = zeros(Num, prod(odims), prod(idims))
    for k in 1:prod(idims)
        m[:, k] = vec(T * reshape(v, idims))
        v = circshift(v, 1)
    end
    return m
end
function Matrix(T::AbstractTransferMatrix, out::Int, s::Array{NTuple{N,Int},K}) where {N,K}
    v = zeros(eltype(T), sum(prod.(s)))
    v[1] = 1
    m = zeros(eltype(T), out, sum(prod.(s)))
    for k in 1:sum(prod.(s))
        m[:, k] = vec(T * BlockBoundaryVector(v, s))
        v = circshift(v, 1)
    end
    return m
end
function blocksizes(dims::Vararg{Vector{Int},N}) where {N}
    [dim for dim in Base.product(dims...)]
end
blocksizes(dims::Vararg{Int,N}) where {N} = blocksizes(([d] for d in dims)...)
function Matrix(T::AbstractTransferMatrix, olength::Int, idims::Dims{2})
    v = vec(zeros(eltype(T), idims))
    v[1] = 1
    m = zeros(eltype(T), olength, prod(idims))
    for k in 1:prod(idims)
        m[:, k] = vec(T * reshape(v, idims))
        v = circshift(v, 1)
    end
    return m
end

# transfer_matrix_bond(mps::AbstractVector{<:OrthogonalLinkSite}, site::Integer, dir::Symbol) = data(link(mps[site], :left))
# transfer_matrix_bond(mps::AbstractVector{<:GenericSite}, site::Integer, dir::Symbol) = TransferMatrix(identity,identity)#Diagonal(I,size(mps[site],1))
# transfer_matrix_bond(mps1::AbstractMPS, mps2::AbstractMPS, site::Integer, dir::Symbol) = kron(transfer_matrix_bond(mps1, site, dir), transfer_matrix_bond(mps2, site, dir))
# transfer_matrix_bond(mps1::AbstractMPS, mpo::AbstractMPO, mps2::AbstractMPS, site::Integer, dir::Symbol) = kron(transfer_matrix_bond(mps1, site, dir), transfer_matrix_bond(mpo, site, dir), transfer_matrix_bond(mps2, site, dir))

# Base.kron(a::UniformScaling, b::UniformScaling) = a * b
# Base.kron(a::UniformScaling, b::AbstractMatrix) = Diagonal(a, size(b, 1)) * b
# Base.kron(a::AbstractMatrix, b::UniformScaling) = Diagonal(b, size(a, 1)) * a

# transfer_matrix_bond(site::OrthogonalLinkSite{T}) where {T} = data(link(site, :left)) #TransferMatrix(v -> data(link(site, :left)) * v, T, (size(site, 1), size(site, 1)))
# transfer_matrix_bond_dense(site::OrthogonalLinkSite) = data(link(site, :left))
# transfer_matrix_bond_dense(site::GenericSite{T}) where {T} = Matrix(one(T)I, size(site, 1))
# transfer_matrix_bond(site::GenericSite{T}) where {T} = I#TransferMatrix(identity, identity, T, (size(site, 1), size(site, 1)))

# transfer_matrix_bond(sites...) = tensor_product(transfer_matrix_bond.(sites)...)

# function transfer_matrix_bond(site::OrthogonalLinkSite{T})
#     f(R) = _transfer_matrix_bond(data(link(site, :left)), R)
#     diag = link(site, :left)
#     @tensor out[:] := diag[-1, 1] * R[1, -2]
#     #TransferMatrix(f,T,Tuple.(size(data(link(site,:left)))))
# end

# _filter_sites(s1::UniformScaling, (scaling, sites)) = (scaling * s1.λ, sites)
# _filter_sites(s1::AbstractMatrix, (scaling, sites)) = (scaling, (s1, sites...))

data(x::UniformScaling) = x.λ
data(x::UniformScaling, ::Symbol) = data(x)

_transfer_matrix_bond(R::Array{<:Number,1}, diags::Vararg{AbstractMatrix,1}) = @tensor out[:] := diags[1][-1, 1] * R[1]
_transfer_matrix_bond(R::Array{<:Number,2}, diags::Vararg{AbstractMatrix,2}) = @tensor out[:] := diags[1][-1, 1] * diags[2][-2, 2] * R[1, 2]
_transfer_matrix_bond(R::Array{<:Number,3}, diags::Vararg{AbstractMatrix,3}) = @tensor out[:] := diags[1][-1, 1] * diags[2][-2, 2] * diags[3][-3, 3] * R[1, 2, 3]
_transfer_matrix_bond(R::Array{<:Number,4}, diags::Vararg{AbstractMatrix,4}) = @tensor out[:] := diags[1][-1, 1] * diags[2][-2, 2] * diags[3][-3, 3] * diags[4][-4, 4] * R[1, 2, 3, 4]

_transfer_matrix_bond(R::Array{<:Number,N}, ds::Vararg{Union{Number,Bool},K}) where {N,K} = prod(ds) * R
_transfer_matrix_bond(R::Array{<:Number,N}, d1::Union{Number,Bool}) where {N} = d1 * R
_transfer_matrix_bond(R::Array{<:Number,2}, d1::AbstractMatrix, d2::Union{Number,Bool}) = @tensor out[:] := d1[-1, 1] * d2 * R[1, -2]
_transfer_matrix_bond(R::Array{<:Number,2}, d1::Union{Number,Bool}, d2::AbstractMatrix) = @tensor out[:] := d2[-2, 2] * d1 * R[-1, 2]
_transfer_matrix_bond(R::Array{<:Number,3}, d1::Union{Number,Bool}, d2::AbstractMatrix, d3::AbstractMatrix) =
    @tensor out[:] := d1 * d2[-2, 2] * d3[-3, 3] * R[-1, 2, 3]
_transfer_matrix_bond(R::Array{<:Number,3}, d1::AbstractMatrix, d2::Union{Number,Bool}, d3::AbstractMatrix) =
    @tensor out[:] := d1[-1, 1] * d2 * d3[-3, 3] * R[1, -2, 3]
_transfer_matrix_bond(R::Array{<:Number,3}, d1::AbstractMatrix, d2::AbstractMatrix, d3::Union{Number,Bool}) =
    @tensor out[:] := d1[-1, 1] * d2[-2, 2] * d3 * R[1, 2, -3]

function transfer_matrix_bond(csites::NTuple{<:Any,Union{GenericSite,OrthogonalLinkSite,MPOsite,LazySiteProduct}}, sites::NTuple{<:Any,Union{GenericSite,OrthogonalLinkSite,MPOsite,LazySiteProduct}})
    K = promote_type(eltype.(sites)...)
    newcsites2 = foldl(_split_lazy, csites, init = ())
    newsites2 = foldr(_split_lazy, sites, init = ())
    cscale, newcsites3 = foldl(_remove_identity, newcsites2, init = (one(K), ()))
    scale, newsites3 = foldr(_remove_identity, newsites2, init = (one(K), ()))
    f(R) = (scale * cscale) * _transfer_matrix_bond(R, data.(link.(newcsites3, :left))..., data.(link.(newsites3, :left))...)
    odims = tuple((size(x, 1) for x in newcsites3)..., (size(x, 1) for x in newsites3)...)
    idims = tuple((size(x)[end] for x in newcsites3)..., (size(x)[end] for x in newsites3)...)
    return TransferMatrix(f, f, K, (odims, idims))
end
# function transfer_matrix_bond(sites::Vararg{Union{OrthogonalLinkSite,GenericSite,MPOsite,ScaledIdentityMPOsite},N}) where {N}
#     diags = data.(link.(sites, :left))
#     # scaling, diags2 = foldr(_filter_sites, diags, init = (1, ()))
#     f(R) = _transfer_matrix_bond(R, diags...)
#     TransferMatrix(f, f, promote_type(eltype.(diags)...), (size.(sites, 1), size.(sites, 1)))
# end

# %% Transfer Matrices
"""
	transfer_left(Γ)

Returns the left transfer matrix of a single site

See also: [`transfer_right`](@ref)
"""

# _transfer_left_mpo(s1::OrthogonalLinkSite, op::MPOsite, s2::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s1, :right), op, GenericSite(s2, :right))
# _transfer_left_mpo(s1::OrthogonalLinkSite, op::MPOsite) = _transfer_left_mpo(GenericSite(s1, :right), op, GenericSite(s1, :right))
# _transfer_left_mpo(s1::OrthogonalLinkSite, s2::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s1, :right), GenericSite(s2, :right))
# _transfer_left_mpo(s::OrthogonalLinkSite) = _transfer_left_mpo(GenericSite(s, :right))

# __transfer_left_mpo(R::AbstractVector,Γ1::GenericSite, Γ2::GenericSite) = __transfer_left_mpo(reshape(R,size(\Gamma)  )))
# __transfer_left_mpo(R::AbstractArray{<:Any,2}, Γ1::GenericSite, Γ2::GenericSite) = (@tensoropt (t1, b1, -1, -2) temp[:] := R[t1, b1] * conj(data(Γ1)[-1, c1, t1]) * data(Γ2)[-2, c1, b1])
# __transfer_left_mpo_adjoint(L::AbstractArray{<:Any,2}, Γ1::GenericSite, Γ2::GenericSite) = @tensoropt (t1, b1, -1, -2) temp[:] := L[t1, b1] * data(Γ1)[t1, c1, -1] * conj(data(Γ2)[b1, c1, -2])
# __transfer_left_mpo(R::AbstractArray{<:Any,3}, Γ1::GenericSite, mpo::MPOsite, Γ2::GenericSite) =
#     @tensoropt (tr, br, -1, -3) temp[:] := conj(data(Γ1)[-1, u, tr]) * data(mpo)[-2, u, d, cr] * data(Γ2)[-3, d, br] * R[tr, cr, br]
# __transfer_left_mpo_adjoint(L::AbstractArray{<:Any,3}, Γ1::GenericSite, mpo::MPOsite, Γ2::GenericSite) =
#     @tensoropt (bl, tl, -1, -3) temp[:] := L[tl, cl, bl] * data(Γ1)[tl, u, -1] * conj(data(mpo)[cl, u, d, -2]) * conj(data(Γ2)[bl, d, -3])

# __transfer_left_mpo(R::AbstractArray{<:Any,3}, Γ1::GenericSite, mpo::ScaledIdentityMPOsite, Γ2::GenericSite) =
#     @tensoropt (t1, b1, -1, -3) temp[:] := data(mpo)*R[t1,-2, b1] * conj(data(Γ1)[-1, c1, t1]) * data(Γ2)[-3, c1, b1]
# __transfer_left_mpo_adjoint(L::AbstractArray{<:Any,3}, Γ1::GenericSite,mpo::ScaledIdentityMPOsite, Γ2::GenericSite) =
#     @tensoropt (t1, b1, -1, -3) temp[:] := L[t1,-2, b1] * data(Γ1)[t1, c1, -1] * conj(data(Γ2)[b1, c1, -3])


__transfer_left_mpo(R::AbstractArray{<:Any,2}, Γ1::Array{<:Any,3}, Γ2::Array{<:Any,3}) = (@tensoropt (t1, b1, -1, -2) temp[:] := R[t1, b1] * Γ1[-1, c1, t1] * Γ2[-2, c1, b1])
__transfer_left_mpo_adjoint(L::AbstractArray{<:Any,2}, Γ1::Array{<:Any,3}, Γ2::Array{<:Any,3}) = @tensoropt (t1, b1, -1, -2) temp[:] := L[t1, b1] * Γ1[t1, c1, -1] * Γ2[b1, c1, -2]
__transfer_left_mpo(R::AbstractArray{<:Any,3}, Γ1::Array{<:Any,3}, mpo::Array{<:Any,4}, Γ2::Array{<:Any,3}) =
    @tensoropt (tr, br, -1, -3) temp[:] := Γ1[-1, u, tr] * mpo[-2, u, d, cr] * Γ2[-3, d, br] * R[tr, cr, br]
__transfer_left_mpo_adjoint(L::AbstractArray{<:Any,3}, Γ1::Array{<:Any,3}, mpo::Array{<:Any,4}, Γ2::Array{<:Any,3}) =
    @tensoropt (bl, tl, -1, -3) temp[:] := L[tl, cl, bl] * Γ1[tl, u, -1] * mpo[cl, u, d, -2] * Γ2[bl, d, -3]

function _transfer_left_mpo(Γ1::NTuple{<:Any,Union{GenericSite,OrthogonalLinkSite,MPOsite}}, Γ2::NTuple{<:Any,Union{GenericSite,OrthogonalLinkSite,MPOsite}})
    K = promote_type(eltype.(Γ1)..., eltype.(Γ2)...)
    g1 = data.(Γ1, :right)
    g2 = data.(Γ2, :right)
    f(R) = __transfer_left_mpo(R, conj.(g1)..., g2...)
    fadj(L) = __transfer_left_mpo_adjoint(L, g1..., conj.(g2)...)
    odims = tuple((size(x, 1) for x in Γ1)..., (size(x, 1) for x in Γ2)...)
    idims = tuple((size(x)[end] for x in Γ1)..., (size(x)[end] for x in Γ2)...)
    return TransferMatrix(f, fadj, K, (odims, idims))
end

# function _transfer_left_mpo(Γ1::Tuple{GenericSite{T}}, Γ2::Tuple{GenericSite{K}}) where {T,K}
#     f(R) = __transfer_left_mpo(R, Γ1, Γ2)
#     fadj(L) = __transfer_left_mpo_adjoint(L, Γ1, Γ2)
#     return TransferMatrix(f, fadj, promote_type(T, K), ((size(Γ1, 1), size(Γ2, 1)), (size(Γ1, 3), size(Γ2, 3))))
# end
# function _transfer_left_mpo(Γ1::GenericSite{T}) where {T}
#     f(R) = __transfer_left_mpo(R, Γ1, Γ1)
#     fadj(L) = __transfer_left_mpo_adjoint(L, Γ1, Γ1)
#     return TransferMatrix(f, fadj, T, ((size(Γ1, 1), size(Γ1, 1)), (size(Γ1, 3), size(Γ1, 3))))
# end
# function _transfer_left_mpo(Γ1::GenericSite{T}, mpo::MPOsite) where {T}
#     f(R) = __transfer_left_mpo(R, Γ1, mpo, Γ1)
#     fadj(L) = __transfer_left_mpo_adjoint(L, Γ1, mpo, Γ1)
#     return TransferMatrix(f, fadj, T, ((size(Γ1, 1), size(mpo, 1), size(Γ1, 1)), (size(Γ1, 3), size(mpo, 4), size(Γ1, 3))))
# end

_transfer_left_mpo(Γ1, mpo::ScaledIdentityMPOsite, Γ2) = data(mpo) * _transfer_left_mpo(Γ1, Γ2)
_transfer_left_mpo(Γ1, mpo::ScaledIdentityMPOsite) = data(mpo) * _transfer_left_mpo(Γ1)
# function _transfer_left_mpo(Γ1::GenericSite{T1}, mpo::MPOsite{T2}, Γ2::GenericSite{T3}) where {T1,T2,T3}
#     f(R) = __transfer_left_mpo(R, Γ1, mpo, Γ2)
#     fadj(L) = __transfer_left_mpo_adjoint(L, Γ1, mpo, Γ2)
#     return TransferMatrix(f, fadj, promote_type(T1, T2, T3), ((size(Γ1, 1), size(mpo, 1), size(Γ2, 1)), (size(Γ1, 3), size(mpo, 4), size(Γ2, 3))))
# end

#TODO Check performance vs ncon, or 'concatenated' versions. Ncon is slower. concatenated is faster
function __transfer_left_mpo(mposites::NTuple{N,MPOsite}) where {N}
    #sizes = size.(mposites)
    rs = size.(mposites, 4)
    ls = size.(mposites, 1)
    #ds = size.(mposites,3)
    us = size.(mposites, 2)
    function contract(R::AbstractArray{<:Number,N})
        site = data(mposites[1])
        # temp = reshape(R, rs[1], prod(rs[2:N])) #0.014480 seconds (250 allocations: 3.682 MiB)
        # @tensor temp[newdone, remaining, down, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

        # temp = reshape(R, rs[1], prod(rs[2:N])) 0.013839 seconds (160 allocations: 3.674 MiB)
        # @tensor temp[down, remaining, newdone, hat] := site[newdone,hat,down,rc] * temp[rc,remaining]

        temp = reshape((R), rs[1], prod(rs[2:N]))
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
    function adjoint_contract(R::AbstractArray{<:Number,N})
        temp = reshape((R), ls[1], prod(ls[2:N]))
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
    map = LinearMapAA(contract, adjoint_contract, (prod(ls), prod(rs));
        idim = rs, odim = ls, T = promote_type(eltype.(mposites)...))
    #LinearMap{promote_type(eltype.(mposites)...)}(contract, adjoint_contract, prod(ls), prod(rs))
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
#::Vararg{Union{AbstractMPOsite,AbstractSite},N}
_transfer_right_mpo(csites::Tuple, sites::Tuple) = _transfer_left_mpo(reverse_direction.(csites), reverse_direction.(sites))
reverse_direction(Γ::Array{<:Number,3}) = permutedims(Γ, [3, 2, 1])
reverse_direction(Γs::AbstractVector{<:Union{AbstractMPOsite,AbstractSite}}) = reverse(reverse_direction.(Γs))
function _transfer_left_gate(Γ1, gate::AbstractSquareGate, Γ2)
    Γnew1 = reverse_direction(Γ1)
    Γnew2 = reverse_direction(Γ2)
    return _transfer_right_gate(Γnew1, reverse_direction(gate), Γnew2)
end

function _transfer_left_gate(Γ, gate::AbstractSquareGate)
    Γnew = reverse_direction(Γ)
    return _transfer_right_gate(Γnew, reverse_direction(gate))
end

_transfer_right_gate(Γ1::AbstractVector{<:AbstractSite}, gate::GenericSquareGate) = _transfer_right_gate(Γ1, gate, Γ1)
function _transfer_right_gate(Γ1::AbstractVector{<:Union{SiteSum,AbstractSite}}, gate::GenericSquareGate, Γ2::AbstractVector{<:Union{SiteSum,AbstractSite}})
    n1 = length(sites(Γ1[1]))
    n2 = length(sites(Γ2[1]))
    Ts = [_transfer_right_gate_dense(getindex.(sites.(Γ1), k1), gate, getindex.(sites.(Γ2), k2)) for (k1, k2) in Base.product(1:n1, 1:n2)]
    return _apply_transfer_matrices(Ts)
end
# function _transfer_right_gate(Γ1::AbstractVector{<:AbstractSite}, gate::GenericSquareGate, Γ2::AbstractVector{<:SiteSum})
#     n1 = length(sites(Γ1[1]))
#     n2 = length(sites(Γ2[1]))
#     Ts = [_transfer_right_gate_dense(getindex.(sites.(Γ1), k1), gate, getindex.(sites.(Γ2), k2)) for (k1, k2) in Base.product(1:n1, 1:n2)]
#     return _apply_transfer_matrices(Ts)
# end

function _transfer_right_gate(Γ1::AbstractVector{<:Union{<:GenericSite,<:OrthogonalLinkSite}}, gate::GenericSquareGate, Γ2::AbstractVector{<:Union{<:GenericSite,<:OrthogonalLinkSite}})
    _transfer_right_gate_dense(Γ1, gate, Γ2)
end

# _transfer_right_gate(Γ1::AbstractVector{<:OrthogonalLinkSite}, gate::GenericSquareGate) = _transfer_right_gate([GenericSite(Γ, :left) for Γ in Γ1], gate)
_transfer_right_gate_dense(Γ1::AbstractVector{<:OrthogonalLinkSite}, gate::GenericSquareGate, Γ2::AbstractVector{<:OrthogonalLinkSite}) = _transfer_right_gate([GenericSite(Γ, :left) for Γ in Γ1], gate, [GenericSite(Γ, :left) for Γ in Γ2])
function _transfer_right_gate_dense(Γ1::AbstractVector{GenericSite{T}}, gate::GenericSquareGate, Γ2::AbstractVector{GenericSite{T}}) where {T}
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
    function T_on_vec(invec::AbstractArray{<:Number,2}) #FIXME Compare performance to a version where the gate is applied between the top and bottom layer of sites
        v = reshape((invec), 1, s_start1, s_start2)
        for k in 1:oplength
            @tensoropt (1, 2) v[:] := conj(data(Γ1[k]))[1, -2, -4] * v[-1, 1, 2] * data(Γ2[k])[2, -3, -5]
            sv = size(v)
            v = reshape(v, prod(sv[1:3]), sv[4], sv[5])
        end
        @tensor vout[:] := v[1, -1, -2] * opvec[1]
        # @tullio vout[a,b] := v[c,a,b] * opvec[c]
        return vout
    end
    #TODO Define adjoint
    return TransferMatrix(T_on_vec, T, ((s_final1, s_final2), (s_start1, s_start2)))#LinearMapAA(T_on_vec, (s_final1 * s_final2, s_start1 * s_start2);
    #odim = (s_final1, s_final2), idim = (s_start1, s_start2), T = T)
end

#Sites 
transfer_matrix(site::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix((site,), (site,), dir)
transfer_matrix(site1::AbstractSite, site2::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix((site1,), (site2,), dir)
transfer_matrix(site::AbstractSite, op::AbstractMPOsite, dir::Symbol = :left) = _local_transfer_matrix((site,), (op, site), dir)
transfer_matrix(site1::AbstractSite, op::AbstractMPOsite, site2::AbstractSite, dir::Symbol = :left) = _local_transfer_matrix((site1,), (op, site2), dir)
transfer_matrix(site::AbstractSite, op::ScaledIdentityGate, dir::Symbol = :left) = data(op) * transfer_matrix(site, dir)
transfer_matrix(site1::AbstractSite, op::ScaledIdentityGate, site2::AbstractSite, dir::Symbol = :left) = data(op) * transfer_matrix(site1, site2, dir)

_purify_site(site::AbstractMPOsite, purify::Bool) = purify ? auxillerate(site) : site
_purify_site(site, purify::Bool) = site
function _local_transfer_matrix(csites::Tuple, sites::Tuple, direction::Symbol)
    K = promote_type(eltype.(sites)...)
    # purify::Bool = any(([ispurification(site) for site in sites if site isa AbstractSite]))
    # newsites::NTuple{<:Any, <:} = Tuple([_purify_site(site,purify) for site in sites if !(site isa ScaledIdentityMPOsite)])
    purify::Bool = any(([ispurification(site) for site in sites if site isa AbstractSite])) || any(([ispurification(site) for site in csites if site isa AbstractSite]))
    newsites = _purify_site.(sites, purify)
    newcsites = _purify_site.(csites, purify)
    newcsites2 = foldl(_split_lazy, newcsites, init = ())
    newsites2 = foldr(_split_lazy, newsites, init = ())
    #scaling = prod([(data(site)) for site in sites if site isa ScaledIdentityMPOsite], init = one(K))
    cscale, newcsites3 = foldl(_remove_identity, newcsites2, init = (one(K), ()))
    scale, newsites3 = foldr(_remove_identity, newsites2, init = (one(K), ()))
    # return (scaling*__local_transfer_matrix(newsites,direction))::LinearMap{K}
    T = (scale * cscale) * __local_transfer_matrix(newcsites3, newsites3, direction)
    return T::CompositeTransferMatrix
end
_split_lazy(s, ss::Tuple) = (s, ss...)
_split_lazy(ss::Tuple, s) = (ss..., s)
_split_lazy(s::LazySiteProduct, ss::Tuple) = (s.sites..., ss...)
_split_lazy(ss::Tuple, s::LazySiteProduct) = (ss..., s.sites...)

_remove_identity(s, (x, ss)::Tuple{<:Number,Tuple}) = (x, (s, ss...))
_remove_identity(s::ScaledIdentityMPOsite, (x, ss)::Tuple{<:Number,Tuple}) = (data(s) * x, ss)

_remove_identity((x, ss)::Tuple{<:Number,Tuple}, s) = (x, (ss..., s))
_remove_identity((x, ss)::Tuple{<:Number,Tuple}, s::ScaledIdentityMPOsite) = (conj(data(s)) * x, ss)

function __local_transfer_matrix(csites::Tuple, sites::Tuple, direction::Symbol = :left)
    if direction == :left
        return _transfer_left_mpo(csites, sites)
    else
        if direction !== :right
            @warn "Defaulting direction to :right"
        end
        return _transfer_right_mpo(csites, sites)
    end
end

_local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::ScaledIdentityGate, direction::Symbol = :left) = data(op) * transfer_matrix(site1, direction)
_local_transfer_matrix(site1::AbstractVector{<:AbstractSite}, op::ScaledIdentityGate, site2::AbstractVector{<:AbstractSite}, direction::Symbol = :left) = data(op) * transfer_matrix(site1, site2, direction)
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
    return [_local_transfer_matrix(sites1[k:k+n-1], op, sites1[k:k+n-1], direction) for k in 1:length(sites1)+1-n]
end

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractMPOsite, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2)
    return [_local_transfer_matrix((sites1[k],), (op, sites2[k]), direction) for k in 1:length(sites1)]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::AbstractMPOsite, direction::Symbol = :left)
    return [_local_transfer_matrix((sites1[k],), (op, sites1[k]), direction) for k in 1:length(sites1)]
end

function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, ops::AbstractVector{<:AbstractSquareGate}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
    @assert length(sites1) == length(sites2) == length(ops)
    N = length(sites1)
    ns = operatorlength.(ops)
    return [_local_transfer_matrix(sites1[k:k+ns[k]-1], op, sites2[k:k+ns[k]-1], direction) for (k, op) in enumerate(ops) if !(k + ns[k] - 1 > N)]
end
function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, ops::AbstractVector{<:AbstractSquareGate}, direction::Symbol = :left)
    @assert length(sites1) == length(ops)
    N = length(sites1)
    # Ts = LinearMap{numtype(sites1)}[]
    ns = operatorlength.(ops)
    return [_local_transfer_matrix(sites1[k:k+ns[k]-1], op, sites1[k:k+ns[k]-1], direction) for (k, op) in enumerate(ops) if !(k + ns[k] - 1 > N)]
end

# function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::Union{AbstractMPO,AbstractVector{<:MPOsite}}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
#     @assert length(sites1) == length(sites2) == length(op)
#     return [_local_transfer_matrix(sos, direction) for sos in zip(sites1, op, sites2)]
# end
# function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, op::Union{AbstractMPO,AbstractVector{<:MPOsite}}, direction::Symbol = :left)
#     @assert length(sites1) == length(op)
#     return [_local_transfer_matrix(so, direction) for so in zip(sites1, op, sites1)]
# end
# function transfer_matrices(sites1::AbstractVector{<:AbstractSite}, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left)
#     @assert length(sites1) == length(sites2)
#     return [_local_transfer_matrix(ss, direction) for (s1,s2) in zip(sites1, sites2)]
# end
# transfer_matrices(sites::AbstractVector{<:AbstractSite}, direction::Symbol = :left) = [_local_transfer_matrix(tuple(site), direction) for site in sites]

function transfer_matrices(csites::NTuple{N1,AbstractVector}, sites::NTuple{N2,AbstractVector}, direction::Symbol) where {N1,N2}
    N = length(csites[1])
    @assert all(N .== length.(csites)) && all(N .== length.(csites)) "Error: different lengths in function transfer_matrices"
    return [_local_transfer_matrix(getindex.(csites, n), getindex.(sites, n), direction) for n in 1:N]
end
function transfer_matrix(sites1::AbstractVector{<:AbstractSite{T}}, op, sites2::AbstractVector{<:AbstractSite}, direction::Symbol = :left) where {T}
    Ts = CompositeTransferMatrix.(transfer_matrices((sites1,), (op, sites2), direction))
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    # if direction == :right
    #     Ts = reverse(Ts)
    # end
    # N = length(Ts)
    # if N > 20
    #     @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    # end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return CompositeTransferMatrix{T}(Tuple(Ts))
    #return foldr(*, Ts)::CompositeTransferMatrix{<:NTuple{<:Any,AbstractTransferMatrix{T,<:Any}},T,<:NTuple{2,<:Any}} #Products of many linear operators cause long compile times!
end
function transfer_matrix(sites1::AbstractVector{<:AbstractSite{T}}, direction::Symbol = :left) where {T}
    Ts = transfer_matrices((sites1,), (sites1,), direction)
    N = length(sites1)
    # sites = direction == :left ? sites1[1:N] : reverse(sites1)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = reverse(Ts)
    end
    #init = IdentityTransferMatrix(T, ((1, 1), (1, 1)))
    #return foldl(*,Ts, init = init)::CompositeTransferMatrix{_A,T,_B}
    #_local_transfer_matrix(tuple(site), direction) for site in sites

    #Convert to CompositeTransferMatrix for type stability
    #CompositeTransferMatrix(transfer_matrix(site, direction)), *, sites
    return CompositeTransferMatrix{T}(Tuple(Ts))
    #return mapfoldr(site -> CompositeTransferMatrix(transfer_matrix(site, direction)), *, sites)::CompositeTransferMatrix{<:NTuple{<:Any,AbstractTransferMatrix{T,<:Any}},T,<:NTuple{2,<:Any}}
    #return foldr(*, Ts)::CompositeTransferMatrix{_A,T,_B}
end

function transfer_matrix(sites1::AbstractVector{<:AbstractSite{T}}, sites2::AbstractVector{<:AbstractSite{T}}, direction::Symbol = :left) where {T}
    Ts = transfer_matrices((sites1,), (sites2,), direction)
    N = length(sites1)
    # sites = direction == :left ? sites1[1:N] : reverse(sites1)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = reverse(Ts)
    end
    return CompositeTransferMatrix{T}(Tuple(Ts))
end

function transfer_matrix(sites1::AbstractVector{<:AbstractSite{T}}, op::AbstractMPO, direction::Symbol = :left) where {T}
    Ts = transfer_matrices((sites1,), (op, sites1), direction)
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return CompositeTransferMatrix{T}(Tuple(Ts)) #Products of many linear operators cause long compile times!
end

function transfer_matrix(sites1::AbstractVector{<:AbstractSite{T}}, op::AbstractSquareGate, direction::Symbol = :left) where {T}
    Ts = transfer_matrices(sites1, op, sites1, direction)
    N = length(Ts)
    if N > 20
        @warn "Calculating the product of $N transfer_matrices. Products of many linearmaps may cause long compile times!"
    end
    if direction == :right
        Ts = @view Ts[N:-1:1]
    end
    return CompositeTransferMatrix{T}(Tuple(Ts)) #Products of many linear operators cause long compile times!
end