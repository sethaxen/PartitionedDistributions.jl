module PartitionedDistributions

using Distributions: Distributions
using FillArrays: FillArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, I
using PDMats: PDMats

export conditional, marginal

include("utils.jl")
include("conditional.jl")
include("marginal.jl")

end
