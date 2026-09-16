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

    @testset "LKJ" begin
        @testset "d = 2: off-diagonal marginals equal the joint" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                dist = LKJ(2, T(0.5) + 3 * rand(T))
                x = T.(rand(dist))
                lp = logpdf(dist, x)
                logp_ref = [i == j ? zero(lp) : lp for i in 1:2, j in 1:2]
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "d = 3: off-diagonal marginals match numerical integration" begin
            @testset for T in (Float64, Float32)
                dist = LKJ(3, T(1.5) + 3 * rand(T))
                x = T.(rand(dist))
                logp = pointwise_marginal_logpdfs(dist, x)
                @test eltype(logp) === T
                @test all(iszero, diag(logp))
                @test logp == logp'
                @testset for (i, j) in ((1, 2), (1, 3), (2, 3))
                    @test logp[i, j] ≈ numerical_marginal_logpdf(dist, x, i, j) rtol = 1.0e-4
                end
            end
        end
        @testset "diagonal entries not equal to 1 have log-density -Inf" begin
            dist = LKJ(3, 2.0)
            x = rand(dist)
            x[2, 2] = 0.9
            logp = pointwise_marginal_logpdfs(dist, x)
            @test logp[2, 2] == -Inf
            @test logp[1, 1] == logp[3, 3] == 0
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
            test_pointwise_marginal_matches_marginal(mix_mv, wrap_array(Ar, x))
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
            test_pointwise_marginal_matches_marginal(mix, wrap_array(Ar, x))
        end
    end

    @testset "MixtureModel (matrix-variate components)" begin
        @testset for Ar in (Array, DimArray), T in (Float64, Float32)
            dist_a = MatrixNormal(randn(T, 3, 4), rand_pdmat(PDMat{T}, 3), rand_pdmat(PDMat{T}, 4))
            dist_b = MatrixNormal(randn(T, 3, 4), rand_pdmat(PDMat{T}, 3), rand_pdmat(PDMat{T}, 4))
            mix_mn = MixtureModel([dist_a, dist_b], T[0.45, 0.55])
            x = rand(dist_a)  # rand(mix_mn) is not supported for matrix-variate mixtures
            test_pointwise_marginal_matches_marginal(mix_mn, wrap_array(Ar, x))
        end
    end

    @testset "JointOrderStatistics" begin
        @testset for Ar in (Array, DimArray),
                T in (Float64, Float32),
                udist in [Normal(rand(T)...), Beta(rand(T)...)],
                n in (10, 20),
                ranks in (sort(shuffle(1:n)[1:5]), 1:n, [1, n], [n ÷ 2])

            dist = JointOrderStatistics(udist, n, ranks)
            x = rand(dist)
            test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
        end
    end

    @testset "ProductDistribution" begin
        @testset "ProductDistribution (multivariate components)" begin
            @testset for Ar in (Array, DimArray),
                    TA in (PDMat, PDiagMat, ScalMat),
                    T in (Float64, Float32),
                    d in (3, 4)

                Σ = rand_pdmat(TA{T}, d)
                comp_dists = [MvNormal(randn(T, d), Σ) for _ in 1:3]
                dist = product_distribution(comp_dists)
                x = rand(dist)
                test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
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
                test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
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
                test_pointwise_marginal_matches_marginal(dist, wrap_array(Ar, x))
            end
        end
    end

    @testset "ProductNamedTupleDistribution" begin
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
                logp_nt = pointwise_marginal_logpdfs(d, x)
                @test logp_nt isa NamedTuple
                @test keys(logp_nt) === keys(x)
                @testset for k in keys(logp_nt)
                    @test logp_nt[k] ≈ pointwise_marginal_logpdfs(d.dists[k], x[k])
                end
                logp_nt2 = map(similar, logp_nt)
                out = pointwise_marginal_logpdfs!!(logp_nt2, d, x)
                @test all(map(===, out, logp_nt2))
                @test _isapprox(out, logp_nt)
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
            logp_nt = pointwise_marginal_logpdfs(outer, z)
            @test logp_nt isa NamedTuple
            @test keys(logp_nt) === keys(z)
            @testset for k in keys(logp_nt)
                @test _isapprox(logp_nt[k], pointwise_marginal_logpdfs(outer.dists[k], z[k]))
            end
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
                test_pointwise_marginal_matches_marginal(rdist, wrap_array(Ar, ry))
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
        ref = pointwise_marginal_logpdfs(dist, x)
        out = invoke(
            pointwise_marginal_logpdfs!!,
            Tuple{
                AbstractMatrix{T},
                Distributions.Distribution{Distributions.ArrayLikeVariate{2}},
                AbstractMatrix{T},
            },
            logp,
            dist,
            x,
        )
        @test out === logp
        @test out ≈ ref rtol = cbrt(eps(T))
    end

    @testset "Wishart" begin
        @testset "1 × 1: marginal equals the joint (Gamma)" begin
            @testset for T in (Float64, Float32), df in (T(0.5), T(3))
                s = T(0.3) + abs(randn(T))
                dist = Wishart(df, fill(s, 1, 1))
                x = rand(dist)
                logp_ref = fill(logpdf(dist, x), 1, 1)
                test_pointwise_marginal_matches_reference(dist, x, logp_ref)
                @test logp_ref[1] ≈ logpdf(Gamma(df / 2, 2 * s), x[1]) rtol = default_rtol(dist, 0)
            end
        end
        @testset "2 × 2, df = 2, ρ = 0: diagonal Gamma, off-diagonal Laplace" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                s1, s2 = T(0.3) .+ abs.(randn(T, 2))
                dist = Wishart(2, PDiagMat([s1, s2]))
                x = rand(dist)
                a = sqrt(s1 * s2)
                logp_ref = [
                    logpdf(Gamma(1, 2 * s1), x[1, 1]) logpdf(Laplace(0, a), x[1, 2])
                    logpdf(Laplace(0, a), x[2, 1]) logpdf(Gamma(1, 2 * s2), x[2, 2])
                ]
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "Monte Carlo normalization" begin
            @testset for p in (2, 3), df in (1, p, 4.5, 3000.0)
                dist = Wishart(df, rand_pdmat(PDMat{Float64}, p))
                @test dist.singular == (df <= p - 1)
                test_pointwise_marginal_mc_normalization(dist, 200_000; atol = 0.03)
            end
        end
        @testset "off-diagonal entry equal to zero" begin
            dist = Wishart(4.5, rand_pdmat(PDMat{Float64}, 2))
            x = rand(dist)
            logp = pointwise_marginal_logpdfs(dist, x)
            x0 = copy(x)
            x0[1, 2] = x0[2, 1] = 0
            logp0 = pointwise_marginal_logpdfs(dist, x0)
            @test isfinite(logp0[1, 2])
            @test logp0[1, 1] == logp[1, 1]
            @testset for δ in (1.0e-8, -1.0e-8)
                xδ = copy(x0)
                xδ[1, 2] = xδ[2, 1] = δ
                @test pointwise_marginal_logpdfs(dist, xδ)[1, 2] ≈ logp0[1, 2] rtol = 1.0e-6
            end
        end
    end

    @testset "VonMisesFisher" begin
        # sampling and the unit-vector check only work reliably for Float64 parameters,
        # so construct with Float64 and convert (which skips the check)
        @testset "D = 2: coordinates of (cos θ, sin θ) with θ ~ VonMises" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32)
                θ0 = 2π * rand() - π
                κ = 0.5 + 4 * rand()
                dist64 = VonMisesFisher([cos(θ0), sin(θ0)], κ)
                dist = convert(VonMisesFisher{T}, dist64)
                x = T.(rand(dist64))
                logp_ref = [vonmises_coordinate_logpdf(θ0, κ, i, Float64(x[i])) for i in 1:2]
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
        end
        @testset "D = 3: matches integration over the orthogonal circle" begin
            @testset for Ar in (Array, DimArray), T in (Float64, Float32), κ in (0.5, 5.0)
                dist64 = VonMisesFisher(normalize(randn(3)), κ)
                dist = convert(VonMisesFisher{T}, dist64)
                x = T.(rand(dist64))
                logp_ref = [numerical_marginal_logpdf(dist64, x, i) for i in 1:3]
                test_pointwise_marginal_matches_reference(dist, wrap_array(Ar, x), logp_ref)
            end
            @testset "coordinate equal to ±1 has a finite density" begin
                dist = VonMisesFisher(normalize(randn(3)), 2.0)
                @testset for x in ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0])
                    logp = pointwise_marginal_logpdfs(dist, x)
                    @test all(isfinite, logp)
                    @test logp ≈ [numerical_marginal_logpdf(dist, x, i) for i in 1:3]
                end
            end
        end
        @testset "coordinates outside [-1, 1] have log-density -Inf" begin
            dist = VonMisesFisher(normalize(randn(4)), 2.0)
            logp = pointwise_marginal_logpdfs(dist, [1.5, 0.0, 0.0, -1.0000001])
            @test logp[[1, 4]] == [-Inf, -Inf]
            @test all(isfinite, logp[2:3])
        end
        @testset "Monte Carlo normalization" begin
            @testset for D in (3, 4, 6, 10), κ in (0.5, 5.0, 40.0)
                dist = VonMisesFisher(normalize(randn(D)), κ)
                test_pointwise_marginal_mc_normalization(dist, 200_000; atol = 0.03)
            end
        end
        @testset "large D and small κ, where the cached normalizing constant overflows" begin
            dist = VonMisesFisher([1.0; zeros(599)], 5.0)
            x = normalize(randn(600))  # any unit vector is a valid point
            @test all(isfinite, pointwise_marginal_logpdfs(dist, x))
            # the sampler returns NaN for large D in Distributions < 0.25.123, so the Monte
            # Carlo check can only run where sampling works
            if all(isfinite, rand(dist))
                test_pointwise_marginal_mc_normalization(dist, 20_000; atol = 0.05)
            else
                @test_skip test_pointwise_marginal_mc_normalization(dist, 20_000; atol = 0.05)
            end
        end
    end
end
