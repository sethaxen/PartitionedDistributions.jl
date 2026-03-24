using Distributions
using InvertedIndices: Not
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Test

"""
    test_logpdf_decomposition(dist, x, inds, comp_inds)

Test the chain rule identity:

    logpdf(dist, x) ≈ logpdf(conditional(dist, x, inds...), x[inds...]) +
                      logpdf(marginal(dist, comp_inds...), x[comp_inds...])

`inds` and `comp_inds` are tuples of indices into the support of `dist` that together
cover all elements.
"""
function test_logpdf_decomposition(dist, x, inds, comp_inds)
    cond_dist = conditional(dist, x, inds...)
    marg_dist = marginal(dist, comp_inds...)
    return @test logpdf(dist, x) ≈ logpdf(cond_dist, x[inds...]) + logpdf(marg_dist, x[comp_inds...])
end

"""
    test_marginal_moments_match(dist, inds...; test_var::Bool=true, test_cov::Bool=false)

Test that the moments of the marginal distribution match the moments of the original distribution.
"""
function test_marginal_moments_match(dist, inds...; test_var::Bool = true, test_cov::Bool = false)
    return @testset "Marginal moments match" begin
        marg_dist = marginal(dist, inds...)
        @testset "Mean matches" begin
            mean_dist = mean(dist)
            mean_marg = mean(marg_dist)
            @test mean_marg ≈ mean_dist[inds...]
        end
        test_var && @testset "Variance matches" begin
            @test var(marg_dist) ≈ var(dist)[inds...]
        end
        test_cov && @testset "Covariance matches" begin
            # for matrix-variate and higher-dimensional, the covariance returned is that of vec(dist)
            lin_inds = vec(LinearIndices(size(dist))[inds...])
            @test cov(marg_dist) ≈ cov(dist)[lin_inds, lin_inds]
        end
    end
end

# For each index specification i, test both:
#   conditional(dist, y, i)  with marginal(dist, Not(i))
#   conditional(dist, y, Not(i)) with marginal(dist, i)   [swap]
# and test that marginal moments match the moments of the original distribution
function test_all_index_combos(dist, y, index_specs)
    return @testset for i in index_specs
        test_logpdf_decomposition(dist, y, (i,), (Not(i),))
        test_logpdf_decomposition(dist, y, (Not(i),), (i,))
        test_marginal_moments_match(dist, i)
    end
end

# For 2-index distributions (e.g. MatrixNormal), test both directions for each axis separately.
function test_all_index_combos_2d(dist, y, row_specs, col_specs)
    @testset for i1 in row_specs
        test_logpdf_decomposition(dist, y, (i1, :), (Not(i1), :))
        test_logpdf_decomposition(dist, y, (Not(i1), :), (i1, :))
    end
    @testset for i2 in col_specs
        test_logpdf_decomposition(dist, y, (:, i2), (:, Not(i2)))
        test_logpdf_decomposition(dist, y, (:, Not(i2)), (:, i2))
    end
    return @testset for i1 in row_specs, i2 in col_specs
        test_marginal_moments_match(dist, i1, i2; test_cov = true)
    end
end

@testset "conditional-marginal consistency" begin
    @testset "AbstractMvNormal (MvNormal)" begin
        Σ = [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]
        dist = MvNormal([1.0, 2.0, 3.0], Σ)
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(
            dist, y, [
                1,                            # trivial: single int
                3,                            # trivial: last element
                1:2,                          # range
                2:-1:1,                       # reverse range
                [1, 3],                       # int array
                Not(3),                       # Not
                Bool[true, false, true],      # bool array
                :,                            # colon
            ]
        )
    end

    @testset "MvNormalCanon" begin
        J = [2.0 -0.5 0.0; -0.5 2.0 -0.5; 0.0 -0.5 2.0]
        dist = MvNormalCanon([1.0, 2.0, 3.0], J)
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(
            dist, y, [
                1,
                3,
                1:2,
                2:-1:1,
                [1, 3],
                Not(2),
                Bool[false, true, true],
                :,
            ]
        )
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
        test_all_index_combos_2d(
            dist, y,
            [1:2, [1, 2], Not(3), Bool[true, true, false], 1:1],    # row specs
            [1:3, [1, 2, 3], Not(4), Bool[true, true, true, false], 1:1],  # col specs
        )
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
        test_all_index_combos(
            dist, y, [
                1,
                3,
                1:2,
                2:-1:1,
                [1, 3],
                Not(1),
                Bool[true, false, true],
                :,
            ]
        )
    end

    @testset "GenericMvTDist" begin
        Σ = [
            1.0 0.5 0.25
            0.5 1.0 0.5
            0.25 0.5 1.0
        ]
        dist = MvTDist(5.0, [1.0, 2.0, 3.0], PDMat(Symmetric(Σ)))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(
            dist, y, [
                1,
                3,
                1:2,
                [1, 3],
                Not(3),
                Bool[true, false, true],
                :,
            ]
        )
    end

    # PDiagMat and ScalMat covariances: cover _schur_complement_and_factor(PDiagMat/ScalMat, i)
    # and _pdview(PDiagMat/ScalMat, i) — both Int and non-Int branches.
    @testset "GenericMvTDist (PDiagMat)" begin
        dist = Distributions.GenericMvTDist(5.0, [1.0, 2.0, 3.0], PDiagMat([1.0, 2.0, 1.5]))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(
            dist, y, [
                1,
                3,
                1:2,
                [1, 3],
                Not(3),
                Bool[true, false, true],
                :,
            ]
        )
    end

    @testset "GenericMvTDist (ScalMat)" begin
        dist = Distributions.GenericMvTDist(5.0, [1.0, 2.0, 3.0], ScalMat(3, 2.0))
        y = [0.5, 1.5, 2.5]
        test_all_index_combos(
            dist, y, [
                1,
                1:2,
                [1, 3],
                Not(3),
                Bool[true, false, true],
            ]
        )
    end

    if isdefined(Distributions, :ProductDistribution)
        @testset "ProductDistribution{1,0} (scalar components)" begin
            # NOTE: currently product_distribution returns a Product, not a ProductDistribution
            dist = Distributions.ProductDistribution([Normal(k, 1.0) for k in 1:5])
            y = [0.5, 1.5, 2.5, 3.5, 4.5]
            test_all_index_combos(
                dist, y, [
                    1,                                           # trivial: single int
                    5,                                           # trivial: last element
                    1:3,                                         # range
                    [1, 3, 5],                                   # int array
                    Not(1),                                      # Not with Int
                    Not(2:4),                                    # Not with range
                    Bool[true, false, true, false, true],        # bool array
                    :,                                           # colon
                ]
            )
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
        end

        if isdefined(Distributions, :Product)
            @testset "Product (scalar components)" begin
                dist = Distributions.Product([Normal(k, 1.0) for k in 1:5])
                y = [0.5, 1.5, 2.5, 3.5, 4.5]
                test_all_index_combos(
                    dist, y, [
                        1,
                        5,
                        1:3,
                        [1, 3, 5],
                        Not(1),
                        Not(2:4),
                        Bool[true, false, true, false, true],
                        :,
                    ]
                )
            end
        end
    end
end
