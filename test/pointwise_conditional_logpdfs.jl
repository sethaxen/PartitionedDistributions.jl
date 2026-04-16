using Distributions
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Random
using Test

@testset "pointwise_conditional_logpdfs" begin
    @testset "MvNormal" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (3, 4)

            Σ = rand_pdmat(TA{T}, n)
            dist = MvNormal(randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
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
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
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
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
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
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
        end
    end

    @testset "GenericMvTDist" begin
        @testset for Ar in (Array, DimArray), T in (Float64, Float32), n in (3, 4)
            Σ = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MvTDist(ν, randn(T, n), Σ)
            x = rand(dist)
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
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
            test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
        end
    end

    @testset "MixtureModel (multivariate)" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (4, 5)

            Σ_a = rand_pdmat(TA{T}, n)
            Σ_b = rand_pdmat(TA{T}, n)
            mix_mv = MixtureModel(
                [MvNormal(randn(T, n), Σ_a), MvNormal(randn(T, n), Σ_b)],
                T[0.4, 0.6],
            )
            x = rand(mix_mv)
            rtol = T <: Float64 ? 1.0e-6 : 1.0e-4
            test_pointwise_matches_conditional(mix_mv, wrap_array(Ar, x); rtol = rtol)
        end
    end

    @testset "MixtureModel (heterogeneous multivariate component types)" begin
        @testset for Ar in (Array, DimArray),
                TA in (PDMat, PDiagMat, ScalMat),
                T in (Float64, Float32),
                n in (4, 5)

            Σ_a = rand_pdmat(TA{T}, n)
            Σ_b = rand_pdmat(TA{T}, n)
            ν = 5 + 10 * rand(T)
            mix = MixtureModel(
                [MvNormal(randn(T, n), Σ_a), Distributions.GenericMvTDist(ν, randn(T, n), Σ_b)],
                T[0.45, 0.55],
            )
            @test !isconcretetype(eltype(Distributions.components(mix)))
            x = T.(rand(mix))
            test_pointwise_matches_conditional(mix, wrap_array(Ar, x); rtol = cbrt(eps(T)))
        end
    end

    @testset "ReshapedDistribution" begin
        @testset for Ar in (Array, DimArray), T in (Float64, Float32)
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
                test_pointwise_matches_conditional(rdist, wrap_array(Ar, ry))
            end
        end
    end

    if isdefined(Distributions, :ProductDistribution)
        @testset "ProductDistribution (multivariate components)" begin
            @testset for Ar in (Array, DimArray),
                    TA in (PDMat, PDiagMat, ScalMat),
                    T in (Float64, Float32),
                    d in (3, 4)

                Σ = rand_pdmat(TA{T}, d)
                comp_dists = [MvNormal(randn(T, d), Σ) for _ in 1:3]
                dist = product_distribution(comp_dists)
                x = rand(dist)
                test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
            end
        end

        @testset "ProductDistribution (scalar components, M == 0)" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32), sz in ((5,), (2, 3))
                ax = map(Base.OneTo, sz)
                factors = map(Iterators.product(ax...)) do _
                    Normal(randn(T), abs(randn(T)))
                end
                # currently, calling product_distribution might produce a Product
                dist = Distributions.ProductDistribution(factors)
                x = rand(dist)
                test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
            end
        end
    end

    if isdefined(Distributions, :Product)
        @testset "Product (univariate factors)" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                dist = Distributions.Product(
                    [Normal(randn(T), T(0.3) + abs(randn(T))) for _ in 1:7],
                )
                x = rand(dist)
                test_pointwise_matches_conditional(dist, wrap_array(Ar, x))
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
            @testset for x in (z, reverse(z))
                logp_nt = pointwise_conditional_logpdfs(d, x)
                @test logp_nt isa NamedTuple
                @test keys(logp_nt) === keys(x)
                @testset for k in keys(logp_nt)
                    @test logp_nt[k] ≈ pointwise_conditional_logpdfs(d.dists[k], x[k])
                end
            end
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
            logp_nt = pointwise_conditional_logpdfs(outer, z)
            @test logp_nt isa NamedTuple
            @test keys(logp_nt) === keys(z)
            @testset for k in keys(logp_nt)
                @test _isapprox(logp_nt[k], pointwise_conditional_logpdfs(outer.dists[k], z[k]))
            end
        end
    end

    if isdefined(Distributions, :JointOrderStatistics)
        @testset "JointOrderStatistics" begin
            @testset for Ar in (Array, DimArray),
                    T in (Float64, Float32),
                    udist in [Normal(rand(T)...), Beta(rand(T)...)],
                    n in (10, 20),
                    ranks in (sort(shuffle(1:n)[1:5]), 1:n, [1, n], [n ÷ 2])

                dist = JointOrderStatistics(udist, n, ranks)
                x = rand(dist)
                xw = wrap_array(Ar, x)
                if length(ranks) == 1
                    logp = pointwise_conditional_logpdfs(dist, xw)
                    @test axes(logp) == axes(xw)
                    @test only(_pointwise_value_array(logp)) ≈ logpdf(dist, xw)
                else
                    test_pointwise_matches_marginal(dist, xw)
                end
            end
        end
    end

    @testset "generic array-variate fallback using invoke" begin
        T = Float64
        m, n = 2, 3
        M = randn(T, m, n)
        U = rand_pdmat(PDMat{T}, m)
        V = rand_pdmat(PDMat{T}, n)
        dist = MatrixNormal(M, U, V)
        x = rand(dist)
        logp = similar(x, T)
        ref = pointwise_conditional_logpdfs(dist, x)
        out = invoke(
            pointwise_conditional_logpdfs!!,
            Tuple{
                AbstractMatrix{T},
                Distributions.Distribution{Distributions.ArrayLikeVariate{2}},
                AbstractMatrix{T},
            },
            logp,
            dist,
            x,
        )
        @test out ≈ ref rtol = cbrt(eps(T))
    end
end
