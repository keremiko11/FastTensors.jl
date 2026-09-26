module FastTensors

using LinearAlgebra
using Statistics

export Tensor, rand_tensor, zeros_tensor, tensor_norm
export contract, unfold, fold, hosvd

include("types.jl")
include("contractions.jl")
include("decompositions.jl")

end # module FastTensors
