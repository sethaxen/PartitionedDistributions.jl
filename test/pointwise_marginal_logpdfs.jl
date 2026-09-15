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

    @testset "MvNormal" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (3, 4)

            Σ = rand_pdmat(TA{T}, n)
            dist = MvNormal(randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "MvNormalCanon" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (3, 4)

            J = rand_pdmat(TA{T}, n)
            μ_c = randn(T, n)
            dist = MvNormalCanon(μ_c, J)
            x = rand(MvNormal(μ_c, PDMat(Symmetric(inv(Matrix(J))))))
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "MatrixNormal" begin
        @testset for Ar in (Array, DimArray),
                T in (Float64, Float32),
                (m, n) in ((3, 4), (2, 5))

            M = randn(T, m, n)
            U = rand_pdmat(PDMat{T}, m)
            V = rand_pdmat(PDMat{T}, n)
            dist = MatrixNormal(M, U, V)
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end
end
