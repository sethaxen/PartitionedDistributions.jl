using Distributions
using InvertedIndices: Not
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Test

@testset "conditional-marginal consistency" begin
    @testset "Univariate (ArrayLikeVariate{0})" begin
        dist = Normal(0, 1)
        x = fill(0.5)
        test_univariate_arraylike_indexing(dist, x)
        @test_throws ArgumentError marginal(dist, 1:2)
        @test_throws ArgumentError marginal(dist, [1, 1])
        @test_throws ArgumentError conditional(dist, x, 1:2)
        @test_throws ArgumentError conditional(dist, x, [1, 1])
    end

    @testset "AbstractMvNormal (MvNormal)" begin
        Σ = [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]
        dist = MvNormal([1.0, 2.0, 3.0], Σ)
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(dist, y)
        @testset "duplicate indices in vector selector" begin
            @test_throws ArgumentError marginal(dist, [1, 1])
            @test_throws ArgumentError conditional(dist, y, [1, 1])
        end
        @testset "single index argument is Colon" begin
            Σ_i = Matrix{Float64}(I, 3, 3)
            mvn = MvNormal([1.0, 2.0, 3.0], Σ_i)
            yv = mean(mvn)
            @test marginal(mvn, :) === mvn
            @test conditional(mvn, yv, :) === mvn
        end
    end

    @testset "MvNormalCanon" begin
        J = [2.0 -0.5 0.0; -0.5 2.0 -0.5; 0.0 -0.5 2.0]
        dist = MvNormalCanon([1.0, 2.0, 3.0], J)
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(dist, y)
    end

    @testset "MatrixNormal (row/col partition)" begin
        M = [1.0 2 3 4; 5 6 7 8; 9 10 11 12]
        U = PDMat(
            [
                4.0 1.0 0.5
                1.0 3.0 0.5
                0.5 0.5 2.0
            ]
        )
        V = PDMat(
            [
                2.0 0.5 0.25 0.1
                0.5 2.0 0.5 0.25
                0.25 0.5 2.0 0.5
                0.1 0.25 0.5 2.0
            ]
        )
        dist = MatrixNormal(M, U, V)
        y = M + 0.1 .* [1 -2 3 -1; -1 2 -3 1; 2 -1 1 -2]
        row_specs = Any[1:2, [1, 2], Not(3), Bool[true, true, false], 1:1]
        col_specs = Any[1:3, [1, 2, 3], Not(4), Bool[true, true, true, false], 1:1]
        test_axis_aligned_partition_combos(dist, y, (row_specs, col_specs))

        @testset "conditional return type by row/column selector shape" begin
            @test @inferred(conditional(dist, y, 1, 2)) isa Normal
            @test @inferred(marginal(dist, 1, 2)) isa Normal
            @test @inferred(conditional(dist, y, 1, 2:4)) isa MvNormal
            @test @inferred(marginal(dist, 1, 2:4)) isa MvNormal
            @test @inferred(conditional(dist, y, 1:2, 2:4)) isa MatrixNormal
            @test @inferred(marginal(dist, 1:2, 2:4)) isa MatrixNormal
        end
        @testset "selecting one row or column" begin
            @testset for inds in [(1, 2:4), (1:2, 3)]
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

    @testset "MatrixNormal (general submatrix vs MvNormal)" begin
        # When the submatrix complement is L-shaped (not selectable by marginal as a
        # MatrixNormal), we verify consistency against the equivalent MvNormal:
        # vec(X) ~ MvNormal(vec(M), kron(V, U)) for X ~ MatrixNormal(M, U, V).
        m, n = 3, 4
        M = [1.0 2 3 4; 5 6 7 8; 9 10 11 12]
        U = PDMat(
            [
                4 1 0.5
                1 3 0.5
                0.5 0.5 2
            ]
        )
        V = PDMat(
            [
                2 0.5 0.25 0.1
                0.5 2 0.5 0.25
                0.25 0.5 2 0.5
                0.1 0.25 0.5 2
            ]
        )
        dist = MatrixNormal(M, U, V)
        y = M + 0.1 .* [1.0 -2 3 -1; -1 2 -3 1; 2 -1 1 -2]
        mvn_dist = vec(dist)

        @testset for (i1, i2) in [
                (1:2, 1:3),                                                # top-left 2×3
                (2:3, 2:4),                                                # bottom-right 2×3
                (1:1, 2:3),                                                # single row, partial cols
                (1:2, 2:4),                                                # rows 1-2, cols 2-4
                ([1, 3], [2, 4]),                                          # int array indices
                (Not(2), Not(1)),                                          # Not indices
                (Bool[true, false, true], Bool[false, true, true, false]), # bool array indices
            ]
            lin_i = vec(LinearIndices(y)[i1, i2])
            lin_ic = setdiff(LinearIndices(y), lin_i)
            cond_mat = conditional(dist, y, i1, i2)
            cond_mvn = conditional(mvn_dist, vec(y), lin_i)
            # MatrixNormal conditional logpdf matches equivalent MvNormal conditional
            @test logpdf(cond_mat, y[i1, i2]) ≈ logpdf(cond_mvn, vec(y[i1, i2]))
            # MvNormal chain rule holds for both the submatrix and its L-shaped complement
            test_logpdf_decomposition(mvn_dist, vec(y), (lin_i,), (lin_ic,))
            test_marginal_moments_match(dist, i1, i2; test_cov = true)
        end
        # scalar element: marginal(MatrixNormal, Int, Int) covers the iszero(ndims) branch.
        # element (1,2) has column-major linear index (2-1)*m+1 = 4 in the vectorized form.
        @test logpdf(marginal(dist, 1, 2), y[1, 2]) ≈ logpdf(marginal(mvn_dist, (2 - 1) * m + 1), y[1, 2])
    end

    @testset "MvLogNormal" begin
        Σ = [
            1.0 0.5 0.25
            0.5 1.0 0.5
            0.25 0.5 1.0
        ]
        dist = MvLogNormal(MvNormal([0.5, 1.0, 1.5], Σ))
        y = exp.([0.5, 1.0, 1.5])
        test_all_index_combos(dist, y)
    end

    @testset "GenericMvTDist" begin
        Σ = [
            1.0 0.5 0.25
            0.5 1.0 0.5
            0.25 0.5 1.0
        ]
        dist = MvTDist(5.0, [1.0, 2.0, 3.0], PDMat(Symmetric(Σ)))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(dist, y)

        @testset "conditional with multivariate kept indices" begin
            c = conditional(dist, y, 1:2)
            @test c isa Distributions.GenericMvTDist
            @test length(c.μ) == 2
        end
    end

    # PDiagMat and ScalMat covariances: cover _schur_complement_and_factor(PDiagMat/ScalMat, i)
    # and _pdview(PDiagMat/ScalMat, i) — both Int and non-Int branches.
    @testset "GenericMvTDist (PDiagMat)" begin
        dist = Distributions.GenericMvTDist(5.0, [1.0, 2.0, 3.0], PDiagMat([1.0, 2.0, 1.5]))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(dist, y)
    end

    @testset "GenericMvTDist (ScalMat)" begin
        dist = Distributions.GenericMvTDist(5.0, [1.0, 2.0, 3.0], ScalMat(3, 2.0))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(dist, y)
    end

    if isdefined(Distributions, :ProductDistribution)
        @testset "ProductDistribution{3,0} (three batch axes)" begin
            dist_3d = Distributions.ProductDistribution(fill(Normal(0, 1), 2, 2, 2))
            x_3d = rand(dist_3d)
            # insufficient indices:
            @test_throws ArgumentError marginal(dist_3d, 1, 1)
            @test_throws ArgumentError conditional(dist_3d, x_3d, 1, 1)
        end

        @testset "ProductDistribution{1,0} (scalar components)" begin
            # NOTE: currently product_distribution returns a Product, not a ProductDistribution
            dist = Distributions.ProductDistribution([Normal(k, 1.0) for k in 1:5])
            y = [0.5, 1.5, 2.5, 3.5, 4.5]
            test_all_index_combos(dist, y)
            # `:` is omitted from `example_vector_indices` here: `marginal(dist, Not(:))` on
            # ProductDistribution hits the linear-index path with an empty selection.
            test_marginal_moments_match(dist, :)
            @test_throws ArgumentError marginal(dist, [1, 1])
        end

        @testset "ProductDistribution{2,1} (multivariate components)" begin
            Σ = [1.0 0.5; 0.5 1.0]
            comp_dists = [MvNormal(k .+ [0, 0.5], Σ) for k in 1:3]
            dist = product_distribution(comp_dists)
            y = hcat([k .+ [0.1, 0.6] for k in 1:3]...)
            # Colon on within-component dim; batch-dim index specs
            @testset for i2 in [1:2, [1, 3], Not(3), Bool[true, false, true], 1:1]
                test_logpdf_decomposition(dist, y, (:, i2), (:, Not(i2)))
                test_logpdf_decomposition(dist, y, (:, Not(i2)), (:, i2))
            end
            # Single component selected with within-component subset:
            # covers iszero(ndims(selected_dists)) && M != 0 branch of conditional/marginal
            @testset "single component, partial within-component" begin
                cond = conditional(dist, y, 1:1, 2)
                @test logpdf(cond, y[1:1, 2]) ≈ logpdf(conditional(comp_dists[2], y[:, 2], 1:1), y[1:1, 2])
                marg = marginal(dist, 1:1, 2)
                @test logpdf(marg, y[1:1, 2]) ≈ logpdf(marginal(comp_dists[2], 1:1), y[1:1, 2])
            end
            @test_throws ArgumentError marginal(dist, [1, 1])
            test_trailing_singleton_indices(dist, y)
            test_multidim_linear_index_matrix_consistency(dist, y)
        end

        @testset "ProductDistribution{2,0} (batch grid of scalars)" begin
            grid = Distributions.ProductDistribution(fill(Normal(0, 1), 2, 2))
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

        if isdefined(Distributions, :Product)
            @testset "Product (scalar components)" begin
                dist = Distributions.Product([Normal(k, 1.0) for k in 1:5])
                y = [0.5, 1.5, 2.5, 3.5, 4.5]
                test_all_index_combos(dist, y)
                @test_throws ArgumentError marginal(dist, [1, 1])
            end
        end
    end

    @testset "MixtureModel (multivariate components)" begin
        Σ_2 = Matrix{Float64}(I, 2, 2)
        mix_mv = MixtureModel(
            [MvNormal(zeros(2), Σ_2), MvNormal(ones(2), Σ_2)],
            [0.4, 0.6],
        )
        y = rand(first(Distributions.components(mix_mv)))
        test_all_index_combos(mix_mv, y)
        @test @inferred(marginal(mix_mv, 1)) isa MixtureModel
        @test @inferred(conditional(mix_mv, y, 1)) isa MixtureModel
    end

    @testset "MixtureModel (matrix-variate components)" begin
        # Distributions.jl does not implement `logpdf(::MixtureModel{Matrixvariate}, ::AbstractMatrix)`;
        # skip full decomposition/moment sweeps here (see multivariate block above).
        M_m = [1.0 2 3 4; 5 6 7 8; 9 10 11 12]
        U_m = PDMat([4.0 1.0 0.5; 1.0 3.0 0.5; 0.5 0.5 2.0])
        V_m = PDMat([2.0 0.5 0.25 0.1; 0.5 2.0 0.5 0.25; 0.25 0.5 2.0 0.5; 0.1 0.25 0.5 2.0])
        dist_mn = MatrixNormal(M_m, U_m, V_m)
        mix_mn = MixtureModel([dist_mn, dist_mn], [0.5, 0.5])
        y = rand(first(Distributions.components(mix_mn)))
        @test @inferred(marginal(mix_mn, 1:2, 2:3)) isa MixtureModel
        @test @inferred(conditional(mix_mn, y, 1:2, 2:3)) isa MixtureModel
    end

    if isdefined(Distributions, :JointOrderStatistics)
        @testset "JointOrderStatistics" begin
            n = 20
            ranks = [3, 7, 11]
            jos = Distributions.JointOrderStatistics(Normal(0.0, 1.0), 20, [3, 7, 11])
            dmarg = @inferred(marginal(jos, 2))
            @test dmarg isa Distributions.OrderStatistic
            @test dmarg.n == n
            @test dmarg.rank == ranks[2]

            dmarg = @inferred(marginal(jos, [1, 3]))
            @test dmarg isa Distributions.JointOrderStatistics
            @test dmarg.n == n
            @test dmarg.ranks == ranks[[1, 3]]
        end
    end
end
