using Distributions
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Random
using Test

@testset "pointwise_marginal_logpdfs" begin
    @testset "Univariate" begin
        @testset for T in (Float64, Float32),
                dist in (Normal(randn(T), T(0.25) + abs(randn(T))), Gamma(T(2), T(3)), Poisson(T(3)))

            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, x)
        end
    end
end
