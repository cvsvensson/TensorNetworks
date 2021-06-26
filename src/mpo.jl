struct MPOsite{T<:Number} <: AbstractArray{T,4}
    data::Array{T,4}
end
function MPOsite(op::Array{T,2}) where {T}
    sop = size(op)
    return MPOsite(reshape(op,1,sop[1],sop[2],1))
end
struct MPO{T<:Number} <: AbstractArray{MPOsite{T}, 1}
    data::Array{MPOsite{T},1}
end
MPO(mpo::MPOsite) = MPO([mpo])
MPO(op::Array{T,2}) where {T} = MPO(MPOsite(op))
MPO(ops::Array{T,1}) where {T} = MPO(map(op->MPOsite(op),ops))
Base.size(mposite::MPOsite) = size(mposite.data)
Base.size(mpo::MPO) = size(mpo.data)
Base.IndexStyle(::Type{<:MPOsite}) = IndexLinear()
Base.IndexStyle(::Type{<:MPO}) = IndexLinear()
Base.getindex(mpo::MPOsite, i::Int) = mpo.data[i]
Base.getindex(mpo::MPO, i::Int) = mpo.data[i]
Base.setindex!(mpo::MPOsite, v, I::Vararg{Int,4}) = (mpo.data[I] = v)
Base.setindex!(mpo::MPO, v, I::Vararg{Int,N}) where {N} = (mpo.data[I] = v)


# %% Todo
"""
gives the mpo corresponding to a*mpo1 + b*mpo2.
"""
function addmpos(mpo1,mpo2,a,b,Dmax,tol=0)
    L = length(mpo1)
    d = size(mpo1[1])[2]
    mpo = Array{Array{Complex{Float64}}}(L)
    mpo[1] = permutedims(cat(1,permutedims(a*mpo1[1],[4,1,2,3]),permutedims(b*mpo2[1],[4,1,2,3])),[2,3,4,1])
    for i = 2:L-1
        mpo[i] = permutedims([permutedims(mpo1[i],[1,4,2,3]) zeros(size(mpo1[i])[1],size(mpo2[i])[4],d,d); zeros(size(mpo2[i])[1],size(mpo1[i])[4],d,d) permutedims(mpo2[i],[1,4,2,3])],[1,3,4,2])
        if tol>0 || size(mpo1[i])[3]+size(mpo2[i])[3] > Dmax
            @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[i-1][-1,-2,-3,1]*mpo[i][1,-4,-5,-6]
            tmp = reshape(tmp,size(mpo[i-1])[1]*d*d,d*d*size(mpo[i])[4])
            F = svd(tmp)
            U,S,V = truncate_svd(F,Dmax,tol)
            mpo[i-1] = reshape(1/2*U*diagm(S),size(mpo[i-1])[1],d,d,D)
            mpo[i] = reshape(2*V,D,d,d,size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1,permutedims(mpo1[L],[1,4,2,3]),permutedims(mpo2[L],[1,4,2,3])),[1,3,4,2])
    if tol>0
        @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[L-1][-1,-2,-3,1]*mpo[L][1,-4,-5,-6]
        tmp = reshape(tmp,size(mpo[L-1])[1]*d*d,d*d*size(mpo[L])[4])
        F = svd(tmp)
        U,S,V = truncate_svd(F,D,tol)
        mpo[L-1] = reshape(1/2*U*diagm(S),size(mpo[L-1])[1],d,d,D)
        mpo[L] = reshape(2*V,D,d,d,size(mpo[L])[4])
    end
    return mpo
end



"""
```multiplyMPOs(mpo1,mpo2; c=true)```
"""
function multiplyMPOs(mpo1,mpo2; c=true)
    L = length(mpo1)
    mpo = Array{Any}(L)
    for j=1:L
        if c
            @tensor temp[:] := mpo1[j][-1,-3,1,-5] * conj(mpo2[j][-2,-4,1,-6])
        else
            @tensor temp[:] := mpo1[j][-1,-3,1,-5] * mpo2[j][-2,1,-4,-6]
        end
        s=size(temp)
        mpo[j] = reshape(temp,s[1]*s[2],s[3],s[4],s[5]*s[6])
    end

    return mpo
end
