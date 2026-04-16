using Distributions
using InvertedIndices: Not
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Random
using Test

@testset "conditional-marginal consistency" begin
    @testset "Univariate (ArrayLikeVariate{0})" begin
        @testset for T in (Float64, Float32)
            dist = Normal(randn(T), T(0.25) + abs(randn(T)))
            x = fill(randn(T))
            test_univariate_arraylike_indexing(dist, x)
            @test_throws ArgumentError marginal(dist, 1:2)
            @test_throws ArgumentError marginal(dist, [1, 1])
            @test_throws ArgumentError conditional(dist, x, 1:2)
            @test_throws ArgumentError conditional(dist, x, [1, 1])
        end
    end

    @testset "AbstractMvNormal (MvNormal)" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 5)
            Σ = rand_pdmat(TA{T}, n)
            dist = MvNormal(randn(T, n), Σ)
            y = rand(dist)
            test_all_index_combos(dist, y)
            @test_throws ArgumentError marginal(dist, [1, 1])
            @test_throws ArgumentError conditional(dist, y, [1, 1])
            @test marginal(dist, :) === dist
            @test conditional(dist, y, :) === dist
        end
    end

    @testset "MvNormalCanon" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 5)
            J = rand_pdmat(TA{T}, n)
            μ_c = randn(T, n)
            dist = MvNormalCanon(μ_c, J)
            y = rand(MvNormal(μ_c, PDMat(Symmetric(inv(Matrix(J))))))
            test_all_index_combos(dist, y)
        end
    end

    @testset "MatrixNormal (row/col partition)" begin
        @testset for T in (Float64, Float32), (m, n) in ((3, 4), (2, 5))
            M = randn(T, m, n)
            U = rand_pdmat(PDMat{T}, m)
            V = rand_pdmat(PDMat{T}, n)
            dist = MatrixNormal(M, U, V)
            y = rand(dist)
            axis_specs = default_axis_specs(dist)
            test_axis_aligned_partition_combos(dist, y, axis_specs)

            @testset "conditional return type by row/column selector shape" begin
                n >= 4 || continue
                @test @inferred(conditional(dist, y, 1, 2)) isa Normal
                @test @inferred(marginal(dist, 1, 2)) isa Normal
                @test @inferred(conditional(dist, y, 1, 2:4)) isa MvNormal
                @test @inferred(marginal(dist, 1, 2:4)) isa MvNormal
                m >= 2 || continue
                @test @inferred(conditional(dist, y, 1:2, 2:4)) isa MatrixNormal
                @test @inferred(marginal(dist, 1:2, 2:4)) isa MatrixNormal
            end
            @testset "selecting one row or column" begin
                ind_pairs = Tuple{Any, Any}[]
                n >= 4 && push!(ind_pairs, (1, 2:4))
                m >= 2 && n >= 3 && push!(ind_pairs, (1:2, 3))
                isempty(ind_pairs) && continue
                @testset for inds in ind_pairs
                    lin_inds = LinearIndices(y)[inds...]
                    dmarg = marginal(dist, inds...)
                    @test dmarg isa MvNormal
                    @test mean(dmarg) ≈ mean(dist)[inds...]
                    @test cov(dmarg) ≈ cov(dist)[lin_inds, lin_inds]
                    dcond = conditional(dist, y, inds...)
                    dcond_lin = conditional(vec(dist), vec(y), lin_inds)
                    @test dcond isa MvNormal
                    @test mean(dcond) ≈ mean(dcond_lin)
                    @test cov(dcond) ≈ cov(dcond_lin)
                end
            end
        end
    end

    @testset "MatrixNormal (general submatrix vs MvNormal)" begin
        # vec(X) ~ MvNormal(vec(M), kron(V, U)); indices assume m == 3, n == 4.
        @testset for T in (Float64, Float32)
            m, n = 3, 4
            M = randn(T, m, n)
            U = rand_pdmat(PDMat{T}, m)
            V = rand_pdmat(PDMat{T}, n)
            dist = MatrixNormal(M, U, V)
            y = rand(dist)
            mvn_dist = vec(dist)

            @testset for (i1, i2) in [
                    (1:2, 1:3),
                    (2:3, 2:4),
                    (1:1, 2:3),
                    (1:2, 2:4),
                    ([1, 3], [2, 4]),
                    (Not(2), Not(1)),
                    (Bool[true, false, true], Bool[false, true, true, false]),
                ]
                lin_i = vec(LinearIndices(y)[i1, i2])
                lin_ic = setdiff(LinearIndices(y), lin_i)
                cond_mat = conditional(dist, y, i1, i2)
                cond_mvn = conditional(mvn_dist, vec(y), lin_i)
                @test logpdf(cond_mat, y[i1, i2]) ≈ logpdf(cond_mvn, vec(y[i1, i2]))
                test_logpdf_decomposition(mvn_dist, vec(y), (lin_i,), (lin_ic,))
                test_marginal_moments_match(dist, i1, i2; test_cov = true)
            end
            @test logpdf(marginal(dist, 1, 2), y[1, 2]) ≈
                logpdf(marginal(mvn_dist, (2 - 1) * m + 1), y[1, 2])
        end
    end

    @testset "MvLogNormal" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (3, 5)
            Σ = rand_pdmat(TA{T}, n)
            dist = MvLogNormal(MvNormal(randn(T, n), Σ))
            y = rand(dist)
            test_all_index_combos(dist, y)
        end
    end

    @testset "GenericMvTDist" begin
        @testset for T in (Float64, Float32), n in (3, 4)
            Σ = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MvTDist(ν, randn(T, n), Σ)
            y = rand(dist)
            test_all_index_combos(dist, y)
            c = conditional(dist, y, 1:2)
            @test c isa Distributions.GenericMvTDist
            @test length(c.μ) == 2
        end
    end

    @testset "MatrixTDist (row/col partition)" begin
        @testset for T in (Float64, Float32), (m, n) in ((3, 4), (2, 5))
            M = randn(T, m, n)
            Σ = rand_pdmat(PDMat{T}, m)
            Ω = rand_pdmat(PDMat{T}, n)
            ν = 5 + 10 * rand(T)
            dist = MatrixTDist(ν, M, Σ, Ω)
            y = rand(dist)
            axis_specs = default_axis_specs(dist)
            test_axis_aligned_partition_combos(dist, y, axis_specs)
            @testset "conditional return type by row/column selector shape" begin
                AffineTDist{T} = LocationScale{T, Continuous, TDist{T}}
                n >= 4 || continue
                @test @inferred(conditional(dist, y, 1, 2)) isa AffineTDist
                @test @inferred(marginal(dist, 1, 2)) isa AffineTDist
                @test @inferred(conditional(dist, y, 1, 2:4)) isa Distributions.GenericMvTDist
                @test @inferred(marginal(dist, 1, 2:4)) isa Distributions.GenericMvTDist
                m >= 2 || continue
                @test @inferred(conditional(dist, y, 1:2, 2:4)) isa Distributions.MatrixTDist
                @test @inferred(marginal(dist, 1:2, 2:4)) isa Distributions.MatrixTDist
            end

            @testset "selecting one row or column" begin
                ind_pairs = Tuple{Any, Any}[]
                n >= 4 && push!(ind_pairs, (1, 2:4))
                m >= 2 && n >= 3 && push!(ind_pairs, (1:2, 3))
                isempty(ind_pairs) && continue
                @testset for inds in ind_pairs
                    lin_inds = LinearIndices(y)[inds...]
                    dmarg = marginal(dist, inds...)
                    @test dmarg isa Distributions.GenericMvTDist
                    dcond = conditional(dist, y, inds...)
                    dcond_lin = conditional(vec(dist), vec(y), lin_inds)
                    @test dcond isa Distributions.GenericMvTDist
                    @test logpdf(dcond_lin, vec(y)[lin_inds]) ≈ logpdf(dcond, y[inds...])
                end
            end
        end
    end

    @testset "ReshapedDistribution" begin
        @testset for T in (Float64, Float32)
            M = randn(T, 3, 4)
            Σ = rand_pdmat(PDMat{T}, 3)
            Ω = rand_pdmat(PDMat{T}, 4)
            ν = T(5) + 10 * rand(T)
            dist = MatrixTDist(ν, M, Σ, Ω)
            y = rand(dist)
            # linear indices into the underlying 3×4 data that factor into per-dim selectors
            col1_inds = vec(LinearIndices(y)[:, 1])  # [1, 2, 3]
            row1_inds = vec(LinearIndices(y)[1, :])  # [1, 4, 7, 10]
            @testset for sz in ((12,), (1, 12), (12, 1), (6, 2), (1, 3, 4))
                rdist = reshape(dist, sz)
                rdist isa Distributions.ReshapedDistribution || continue
                ry = reshape(y, sz)
                test_logpdf_decomposition(rdist, ry, (col1_inds,), (complement_linear(ry, col1_inds),))
                test_logpdf_decomposition(rdist, ry, (row1_inds,), (complement_linear(ry, row1_inds),))
                @test logpdf(marginal(rdist, col1_inds), ry[col1_inds]) ≈
                    logpdf(marginal(dist, :, 1), y[:, 1])
                @test logpdf(marginal(rdist, row1_inds), ry[row1_inds]) ≈
                    logpdf(marginal(dist, 1, :), y[1, :])
                @test logpdf(conditional(rdist, ry, col1_inds), ry[col1_inds]) ≈
                    logpdf(conditional(dist, y, :, 1), y[:, 1])
                @test logpdf(conditional(rdist, ry, row1_inds), ry[row1_inds]) ≈
                    logpdf(conditional(dist, y, 1, :), y[1, :])
            end
        end
    end

    @testset "ProductDistribution" begin
        @testset "ProductDistribution{3,0} (three batch axes)" begin
            @testset for T in (Float64, Float32)
                dist_3d = Distributions.ProductDistribution(
                    [
                        Normal(randn(T), T(0.2) + abs(randn(T))) for _ in 1:2, _ in 1:2, _ in 1:2
                    ]
                )
                x_3d = rand(dist_3d)
                @test_throws ArgumentError marginal(dist_3d, 1, 1)
                @test_throws ArgumentError conditional(dist_3d, x_3d, 1, 1)
            end
        end

        @testset "ProductDistribution{1,0} (scalar components)" begin
            @testset for T in (Float64, Float32)
                dist = Distributions.ProductDistribution(
                    [
                        Normal(randn(T), T(0.3) + abs(randn(T))) for _ in 1:5
                    ]
                )
                y = rand(dist)
                test_all_index_combos(dist, y)
                test_marginal_moments_match(dist, :)
                @test_throws ArgumentError marginal(dist, [1, 1])
            end
        end

        @testset "ProductDistribution{2,1} (multivariate components)" begin
            @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), d in (4, 6)
                Σ = rand_pdmat(TA{T}, d)
                comp_dists = [MvNormal(randn(T, d), Σ) for _ in 1:3]
                dist = product_distribution(comp_dists)
                y = rand(dist)
                @testset for i2 in [1:2, [3, 2], Not(3), Bool[true, false, true], 1:1]
                    test_logpdf_decomposition(dist, y, (:, i2), (:, Not(i2)))
                    test_logpdf_decomposition(dist, y, (:, Not(i2)), (:, i2))
                end
                @testset "single component, partial within-component" begin
                    cond = conditional(dist, y, 1:1, 2)
                    @test logpdf(cond, y[1:1, 2]) ≈
                        logpdf(conditional(comp_dists[2], y[:, 2], 1:1), y[1:1, 2])
                    marg = marginal(dist, 1:1, 2)
                    @test logpdf(marg, y[1:1, 2]) ≈
                        logpdf(marginal(comp_dists[2], 1:1), y[1:1, 2])
                end
                @test_throws ArgumentError marginal(dist, [1, 1])
                r = min(3, d)
                lin = LinearIndices(axes(dist))[r, 2]
                @test marginal(dist, lin) isa Normal
                @test logpdf(marginal(dist, lin), y[lin]) ≈
                    logpdf(marginal(comp_dists[2], r), y[r, 2])
                @test conditional(dist, y, lin) isa Normal
                @test logpdf(conditional(dist, y, lin), y[lin]) ≈
                    logpdf(conditional(comp_dists[2], y[:, 2], r), y[lin])
                test_trailing_singleton_indices(dist, y)
                test_multidim_linear_index_matrix_consistency(dist, y)
            end
        end

        @testset "ProductDistribution{2,0} (batch grid of scalars)" begin
            @testset for T in (Float64, Float32)
                grid = Distributions.ProductDistribution(
                    [
                        Normal(randn(T), T(0.25) + abs(randn(T))) for _ in 1:2, _ in 1:2
                    ]
                )
                x_grid = rand(grid)
                m_row = marginal(grid, 1, :)
                @test m_row isa Distributions.AbstractMvNormal
                @test logpdf(m_row, x_grid[1, :]) ≈
                    logpdf(marginal(grid, 1, 1), x_grid[1, 1]) +
                    logpdf(marginal(grid, 1, 2), x_grid[1, 2])
                c_col = conditional(grid, x_grid, :, 1)
                @test c_col isa Distributions.AbstractMvNormal
                @test isfinite(logpdf(c_col, x_grid[:, 1]))
                lin_1 = LinearIndices(axes(grid))[1, 1]
                @test marginal(grid, lin_1) isa Normal
                @test isfinite(logpdf(conditional(grid, x_grid, lin_1), x_grid[lin_1]))
            end
        end

        if isdefined(Distributions, :Product)
            @testset "Product (scalar components)" begin
                @testset for T in (Float64, Float32)
                    dist = Distributions.Product(
                        [
                            Normal(randn(T), T(0.3) + abs(randn(T))) for _ in 1:5
                        ]
                    )
                    y = rand(dist)
                    test_all_index_combos(dist, y)
                    @test_throws ArgumentError marginal(dist, [1, 1])
                end
            end
        end
    end

    @testset "MixtureModel (multivariate components)" begin
        @testset for TA in (PDMat, PDiagMat, ScalMat), T in (Float64, Float32), n in (4, 6)
            Σ_a = rand_pdmat(TA{T}, n)
            Σ_b = rand_pdmat(TA{T}, n)
            mix_mv = MixtureModel(
                [MvNormal(randn(T, n), Σ_a), MvNormal(randn(T, n), Σ_b)],
                T[0.4, 0.6],
            )
            y = rand(mix_mv)
            rtol = T <: Float64 ? 1.0e-6 : 1.0e-4
            test_all_index_combos(mix_mv, y; rtol)
            @test @inferred(marginal(mix_mv, 1)) isa MixtureModel
            @test @inferred(conditional(mix_mv, y, 1)) isa MixtureModel
        end
    end

    @testset "MixtureModel (matrix-variate components)" begin
        @testset for T in (Float64, Float32)
            dist_a = MatrixNormal(
                randn(T, 3, 4),
                rand_pdmat(PDMat{T}, 3),
                rand_pdmat(PDMat{T}, 4),
            )
            dist_b = MatrixNormal(
                randn(T, 3, 4),
                rand_pdmat(PDMat{T}, 3),
                rand_pdmat(PDMat{T}, 4),
            )
            mix_mn = MixtureModel([dist_a, dist_b], T[0.45, 0.55])
            y = rand(dist_a)
            @test @inferred(marginal(mix_mn, 1:2, 2:3)) isa MixtureModel
            @test @inferred(conditional(mix_mn, y, 1:2, 2:3)) isa MixtureModel
        end
    end

    @testset "JointOrderStatistics" begin
        n = 20
        ranks = [3, 7, 11]
        jos = Distributions.JointOrderStatistics(Normal(0, 1), 20, [3, 7, 11])
        dmarg = @inferred(marginal(jos, 2))
        @test dmarg isa Distributions.OrderStatistic
        @test dmarg.n == n
        @test dmarg.rank == ranks[2]

        dmarg = @inferred(marginal(jos, [1, 3]))
        @test dmarg isa Distributions.JointOrderStatistics
        @test dmarg.n == n
        @test dmarg.ranks == ranks[[1, 3]]
    end

    @testset "ProductNamedTupleDistribution" begin
        Σ = [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]
        d = product_distribution(
            (
                x = MvNormal(zeros(3), Σ),
                y = MvNormal([1.0, 2.0], [2.0 0.0; 0.0 3.0]),
            )
        )
        @test marginal(d, :) === d
        @test marginal(d, 1) === d.dists.x
        @test marginal(d, :y) === d.dists.y
        _marginal(d) = @inline marginal(d, (:y, :x))
        dm = @inferred _marginal(d)
        @test dm isa Distributions.ProductNamedTupleDistribution
        @test keys(dm.dists) === (:y, :x)
        @test dm.dists.y === d.dists.y
        @test dm.dists.x === d.dists.x
        dm2 = marginal(d, [:y, :x])
        @test keys(dm2.dists) === (:y, :x)

        _marginal2(d) = marginal(d, (; x = 1:2, y = 2:2))
        dsub = @inferred _marginal2(d)
        @test dsub isa Distributions.ProductNamedTupleDistribution
        @test keys(dsub.dists) === (:x, :y)
        @test dsub.dists.x == marginal(d.dists.x, 1:2)
        @test dsub.dists.y == marginal(d.dists.y, 2:2)

        _marginal3(d) = marginal(d, (; y = 2:2, x = 1:1))
        dperm = @inferred _marginal3(d)
        @test keys(dperm.dists) === (:y, :x)

        z = rand(d)
        test_logpdf_decomposition(d, z, (:x,), (:y,))
        test_logpdf_decomposition(d, z, (:y,), (:x,))
        test_marginal_moments_match(d, :x; test_cov = false)
        test_marginal_moments_match(d, :y; test_cov = false)
        test_marginal_moments_match(d, (:y, :x); test_cov = false)

        @test conditional(d, z, :x) === d.dists.x
        @test conditional(d, z, (:y, :x)) == marginal(d, (:y, :x))
        z_sel = (x = z.x[1:1], y = z.y[2:2])
        _conditional(d) = conditional(d, z, (; x = 1:1, y = 2:2))
        dcond = @inferred _conditional(d)
        @test dcond isa Distributions.ProductNamedTupleDistribution
        @test logpdf(dcond, z_sel) ≈
            logpdf(conditional(d.dists.x, z.x, 1:1), z_sel.x) +
            logpdf(conditional(d.dists.y, z.y, 2:2), z_sel.y)

        @test_throws ArgumentError marginal(d, (;))
        @test_throws Exception marginal(d, (; z = 1))
        @test_throws Exception conditional(d, z, (; z = 1))
        @test_throws DomainError conditional(d, (x = fill(NaN, 3), y = z.y), :x)
    end
end
