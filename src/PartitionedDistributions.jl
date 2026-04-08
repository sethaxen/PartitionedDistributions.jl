module PartitionedDistributions

using Distributions: Distributions
using FillArrays: FillArrays
using InvertedIndices: Not
using LinearAlgebra: LinearAlgebra, I
using LogExpFunctions: LogExpFunctions
using PDMats: PDMats
using StatsBase: StatsBase

export conditional, marginal

include("utils.jl")
include("conditional.jl")
include("marginal.jl")

end
