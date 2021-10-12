struct SqrtDensityMatrix{MPS,Site} <: AbstractMPS{Site}
    mps::MPS

end
#passthrough: canonicalize, 

#QuantumChannel * SqrtDensityMatrix -> Cholesky -> SqrtDensityMatrix

struct DensityMatrix{MPS,Site} <: AbstractMPS{Site}
    mps::MPS

end
# Different canonicalization


#public functions that take operators should always process them.
# However, maybe the user already did so?
# We can introduce different operator types keeping track of this. (Good for quantum channels, but how to make it convenient?)
# Or, label the legs of the operator as physical or auxillary. (But then quantum channels are hard)
function process_operator(op,mps::DensityMatrix)
    #double the operator
end
process_operator(op::QuantumChannel, mps::DensityMatrix) = op
process_operator(op::QuantumChannel, mps::SqrtDensityMatrix) = #process the mps
process_operator(op::QuantumChannel, mps::AbstractMPS) = #process the mps

abstract type AbstractOperator{N,GateOrMPO}
    # Should implement action on N sites -> N sites
    # transfer matrix
    # 
end


struct AuxillaryOp end
struct PhysicalOp end

abstract type AbstractQuantumChannel{N,GateOrMPO} end

struct QuantumChannel{N,GateOrMPO} 
    gateormpo
end


function process_operator(op,mps::SqrtDensityMatrix)

    #auxillerate the operator
end

function process_operator(op,mps::AbstractMPS)

    #Do nothing
end