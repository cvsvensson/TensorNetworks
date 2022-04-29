
_parity_1_plus = 


function _positive_parities(n)
    map(x-> (x >>> -1) + (iseven(count_ones(x)) ? 0 : 1), 0:2^(n-1)-1)
end
struct ParityTensor{N,T} <: AbstractArray{N,T}
    blocks::Vector{Array{N,T}}

end