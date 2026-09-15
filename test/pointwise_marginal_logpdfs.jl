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

    @testset "MvLogNormal" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (3, 4)

            Σ = rand_pdmat(TA{T}, n)
            dist = MvLogNormal(MvNormal(randn(T, n), Σ))
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "GenericMvTDist" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (3, 4)

            Σ = rand_pdmat(TA{T}, n)
            ν = 5 + 10 * rand(T)
            dist = Distributions.GenericMvTDist(ν, randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "MatrixTDist" begin
        @testset for Ar in (Array, DimArray),
                T in (Float64, Float32),
                (m, n) in ((3, 4), (2, 5))

            M = randn(T, m, n)
            Σ = rand_pdmat(PDMat{T}, m)
            Ω = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MatrixTDist(ν, M, Σ, Ω)
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "Dirichlet" begin
        @testset "2 components: marginals equal the joint" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                dist = Dirichlet(T(0.5) .+ 3 * rand(T, 2))
                x = rand(dist)
                logp_ref = fill(logpdf(dist, x), 2)
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "3 components: marginals match numerical integration" begin
            @testset for T in (Float64, Float32)
                dist = Dirichlet(T(1.5) .+ 3 * rand(T, 3))
                x = rand(dist)
                logp = pointwise_marginal_logpdfs(dist, x)
                @test eltype(logp) === T
                @testset for i in 1:3
                    @test logp[i] ≈ numerical_marginal_logpdf(dist, x, i) rtol = 1.0e-4
                end
            end
        end
    end

    @testset "Multinomial" begin
        @testset "2 categories: marginals equal the joint" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                p = rand(T)
                dist = Multinomial(10, [p, 1 - p])
                x = rand(dist)
                logp_ref = fill(logpdf(dist, x), 2)
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "3 categories: marginals match exhaustive summation" begin
            @testset for T in (Float64, Float32)
                p = rand(T, 3)
                p ./= sum(p)
                dist = Multinomial(8, p)
                x = rand(dist)
                logp_ref = [enumerated_marginal_logpdf(dist, x, i) for i in 1:3]
                test_pointwise_marginal_matches_reference(dist, x, logp_ref)
                @test eltype(pointwise_marginal_logpdfs(dist, x)) === T
            end
        end
    end

    @testset "DirichletMultinomial" begin
        @testset "2 categories: marginals equal the joint" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                dist = DirichletMultinomial(10, T(0.5) .+ 3 * rand(T, 2))
                x = rand(dist)
                logp_ref = fill(logpdf(dist, x), 2)
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "3 categories: marginals match exhaustive summation" begin
            @testset for T in (Float64, Float32)
                dist = DirichletMultinomial(8, T(0.5) .+ 3 * rand(T, 3))
                x = rand(dist)
                logp_ref = [enumerated_marginal_logpdf(dist, x, i) for i in 1:3]
                test_pointwise_marginal_matches_reference(dist, x, logp_ref)
                @test eltype(pointwise_marginal_logpdfs(dist, x)) === T
            end
        end
    end
end
