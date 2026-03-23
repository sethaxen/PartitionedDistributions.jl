module PartitionedDistributions

using Distributions: Distributions
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, I

export conditional, marginal

include("conditional.jl")
include("marginal.jl")

end
