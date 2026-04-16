using Distributions
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Random
using Test

@testset "pointwise_conditional_logpdfs" begin
    @testset "MvNormal" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 4)
            Σ = rand_pdmat(TA{T}, n)
            dist = MvNormal(randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "MvNormalCanon" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 4)
            J = rand_pdmat(TA{T}, n)
            μ_c = randn(T, n)
            dist = MvNormalCanon(μ_c, J)
            x = rand(MvNormal(μ_c, PDMat(Symmetric(inv(Matrix(J))))))
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "MatrixNormal" begin
        @testset for T in (Float64, Float32), (m, n) in ((3, 4), (2, 5))
            M = randn(T, m, n)
            U = rand_pdmat(PDMat{T}, m)
            V = rand_pdmat(PDMat{T}, n)
            dist = MatrixNormal(M, U, V)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "MvLogNormal" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 4)
            Σ = rand_pdmat(TA{T}, n)
            dist = MvLogNormal(MvNormal(randn(T, n), Σ))
            x = rand(dist)
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "GenericMvTDist" begin
        @testset for T in (Float64, Float32), n in (3, 4)
            Σ = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MvTDist(ν, randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "MatrixTDist" begin
        @testset for T in (Float64, Float32), (m, n) in ((3, 4), (2, 5))
            M = randn(T, m, n)
            Σ = rand_pdmat(PDMat{T}, m)
            Ω = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MatrixTDist(ν, M, Σ, Ω)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, x)
        end
    end

    @testset "MixtureModel (multivariate)" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (4, 5)
            Σ_a = rand_pdmat(TA{T}, n)
            Σ_b = rand_pdmat(TA{T}, n)
            mix_mv = MixtureModel(
                [MvNormal(randn(T, n), Σ_a), MvNormal(randn(T, n), Σ_b)],
                T[0.4, 0.6],
            )
            x = rand(mix_mv)
            rtol = T <: Float64 ? 1.0e-6 : 1.0e-4
            test_pointwise_matches_conditional(mix_mv, x; rtol = rtol)
        end
    end

    @testset "ReshapedDistribution" begin
        @testset for T in (Float64, Float32)
            m, n = 3, 4
            M = randn(T, m, n)
            U = rand_pdmat(PDMat{T}, m)
            V = rand_pdmat(PDMat{T}, n)
            dist = MatrixNormal(M, U, V)
            y = rand(dist)
            for sz in ((n, m), (m * n,))
                rdist = reshape(dist, sz)
                rdist isa Distributions.ReshapedDistribution || continue
                ry = reshape(y, sz)
                test_pointwise_matches_conditional(rdist, ry)
            end
        end
    end

    if isdefined(Distributions, :ProductDistribution)
        @testset "ProductDistribution (multivariate components)" begin
            @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), d in (3, 4)
                Σ = rand_pdmat(TA{T}, d)
                comp_dists = [MvNormal(randn(T, d), Σ) for _ in 1:3]
                dist = product_distribution(comp_dists)
                x = rand(dist)
                test_pointwise_matches_conditional(dist, x)
            end
        end
    end

    if isdefined(Distributions, :Product)
        @testset "Product (univariate factors)" begin
            @testset for T in (Float64, Float32)
                dist = Distributions.Product(
                    [Normal(randn(T), T(0.3) + abs(randn(T))) for _ in 1:7],
                )
                x = rand(dist)
                test_pointwise_matches_conditional(dist, x)
            end
        end
    end

    if isdefined(Distributions, :ProductNamedTupleDistribution)
        @testset "ProductNamedTupleDistribution" begin
            Σ = [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]
            d = product_distribution(
                (
                    x = MvNormal(zeros(3), Σ),
                    y = MvNormal([1.0, 2.0], [2.0 0.0; 0.0 3.0]),
                )
            )
            z = rand(d)
            test_pointwise_matches_conditional(d, z)
        end

        @testset "nested ProductNamedTuple (inner product + scalars)" begin
            inner = product_distribution(
                (
                    u = MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)),
                    v = Normal(0.5, 0.25),
                )
            )
            outer = product_distribution((block = inner, w = Gamma(2.0, 3.0)))
            z = rand(outer)
            test_pointwise_matches_conditional(outer, z)
        end

        @testset "nested ProductNamedTuple (inner Product vector)" begin
            inner = Distributions.Product(
                [Normal(randn(), 0.4 + abs(randn())) for _ in 1:4],
            )
            outer = product_distribution((chain = inner, tag = Normal(0.2, 0.5)))
            z = rand(outer)
            test_pointwise_matches_conditional(outer, z)
        end
    end

    if isdefined(Distributions, :JointOrderStatistics)
        @testset "JointOrderStatistics" begin
            @testset for T in (Float64, Float32),
                    udist in [Normal(rand(T)...), Beta(rand(T)...)],
                    n in (10, 20),
                    ranks in (sort(shuffle(1:n)[1:5]), 1:n, [1, n])

                dist = JointOrderStatistics(udist, n, ranks)
                x = rand(dist)
                test_pointwise_matches_marginal(dist, x)
            end
        end
    end
end
