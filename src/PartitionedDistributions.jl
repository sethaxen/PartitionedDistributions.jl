module PartitionedDistributions

using Distributions: Distributions
using FillArrays: FillArrays
using InvertedIndices: Not
using IrrationalConstants: logπ, log2π
using LinearAlgebra: LinearAlgebra, I
using LogExpFunctions: LogExpFunctions
using PDMats: PDMats
using SpecialFunctions: SpecialFunctions
using StatsBase: StatsBase

export conditional, marginal
export pointwise_conditional_logpdfs, pointwise_conditional_logpdfs!!

include("utils.jl")
include("conditional.jl")
include("marginal.jl")
include("pointwise_conditional_logpdfs.jl")

end
