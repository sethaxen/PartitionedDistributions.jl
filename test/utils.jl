using Distributions
using FillArrays
using InvertedIndices: Not
using LinearAlgebra
using PartitionedDistributions
using PDMats: PDMat, PDiagMat, ScalMat
using Random
using Test

ref_schur_via_inv(Σ, i::Int) = inv(inv(Σ)[i, i])
ref_schur_via_inv(Σ, i::AbstractVector{Int}) = inv(inv(Σ)[i, i])

index_complement(n::Int, i) = setdiff(1:n, i)

@testset "utils" begin
    @testset "_schur_complement_and_factor" begin
        @testset for TA in [Matrix, PDMat, PDiagMat, ScalMat], T in [Float64, Float32], n in [4, 6]
            A = rand_pdmat(TA{T}, n)
            @test A isa TA{T}

            @testset "Int index" begin
                i = rand(1:n)
                ic = index_complement(n, i)
                S, B, Σ_ic = @inferred PartitionedDistributions._schur_complement_and_factor(A, i)
                @test S ≈ ref_schur_via_inv(A, i)
                @test A[i:i, ic] ≈ B' * A[ic, ic]
                @test Σ_ic ≈ A[ic, ic]
            end

            @testset "Vector index" begin
                k = 3
                i2 = shuffle(1:n)[1:k]
                ic = index_complement(n, i2)
                S, B, Σ_ic = @inferred PartitionedDistributions._schur_complement_and_factor(A, i2)
                @test S ≈ ref_schur_via_inv(A, i2)
                @test A[i2, ic] ≈ B' * A[ic, ic]
                @test Σ_ic ≈ A[ic, ic]
            end
        end
    end

    @testset "_pdview" begin
        @testset for TA in [PDMat, PDiagMat, ScalMat], T in [Float64, Float32], n in [4, 6]
            A = rand_pdmat(TA{T}, n)
            @test A isa TA{T}

            i = shuffle(1:n)[1:3]
            Ai_view = @inferred PartitionedDistributions._pdview(A, i)
            @test Ai_view isa TA{T}
            @test Ai_view ≈ A[i, i]
        end
    end

    @testset "_mvnormal" begin
        @testset for TA in [PDMat, PDiagMat, ScalMat], T in [Float64, Float32], n in [3, 5]
            μ = randn(n)
            Σ = rand_pdmat(TA{T}, n)
            mvn = MvNormal(μ, Σ)
            @test PartitionedDistributions._mvnormal(mvn) === mvn

            J = inv(Σ)
            h = J * μ
            canon = MvNormalCanon(h, J)
            m = @inferred PartitionedDistributions._mvnormal(canon)
            @test m isa MvNormal
            @test mean(m) ≈ μ rtol = cbrt(eps(T))
            @test cov(m) ≈ Σ rtol = cbrt(eps(T))
        end
    end

    @testset "_validate_indices / _validate_index" begin
        x = randn(3, 2)
        cart = CartesianIndices(x)
        loginds = only(Base.to_indices(x, (x .> -Inf,)))
        @test loginds isa Base.LogicalIndex

        valid_ids = [
            cart,
            only(Base.to_indices(cart, (Not(2),))),
            loginds,
            Base.Slice(eachindex(x)),
            2,
            [1, 3, 2],
        ]

        invalid_ids = [
            [1, 2, 1],
        ]

        @testset for id in valid_ids
            PartitionedDistributions._validate_index(id)
        end
        PartitionedDistributions._validate_indices(valid_ids)

        @testset for id in invalid_ids
            @test_throws ArgumentError PartitionedDistributions._validate_index(id)
            @test_throws ArgumentError PartitionedDistributions._validate_indices(vcat(valid_ids, [id]))
        end
        @test_throws ArgumentError PartitionedDistributions._validate_indices(invalid_ids)
    end

    @testset "factorize_indices" begin
        @testset "vector" begin
            A = randn(5)
            inds = shuffle(1:5)[1:3]
            @test PartitionedDistributions.factorize_indices(A, inds) == (inds,)
        end
        @testset "matrix" begin
            A = randn(5, 4)
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 5]) == ([1, 3, 5], [1])
            @test PartitionedDistributions.factorize_indices(A, [5, 3, 1]) == ([5, 3, 1], [1])
            @test PartitionedDistributions.factorize_indices(A, [1, 6]) == ([1], [1, 2])
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 5, 6, 8, 10]) == ([1, 3, 5], [1, 2])
            @test PartitionedDistributions.factorize_indices(A, [6, 8, 10, 1, 3, 5]) == ([1, 3, 5], [2, 1])
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 6, 5, 8, 10]) === nothing
            @test PartitionedDistributions.factorize_indices(A, [2, 4, 6]) === nothing
            inds = LinearIndices(A) .<= 10
            @test PartitionedDistributions.factorize_indices(A, inds) == ([1, 2, 3, 4, 5], [1, 2])
            inds = mod.(LinearIndices(A) .+ 3, 5) .> 2
            @test PartitionedDistributions.factorize_indices(A, inds) == ([1, 5], [1, 2, 3, 4])
        end

        @testset "3D array" begin
            A = randn(5, 4, 3)
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 5]) == ([1, 3, 5], [1], [1])
            @test PartitionedDistributions.factorize_indices(A, [5, 3, 1]) == ([5, 3, 1], [1], [1])
            @test PartitionedDistributions.factorize_indices(A, [1, 6]) == ([1], [1, 2], [1])
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 5, 6, 8, 10]) == ([1, 3, 5], [1, 2], [1])
            @test PartitionedDistributions.factorize_indices(A, [6, 8, 10, 1, 3, 5]) == ([1, 3, 5], [2, 1], [1])
            @test PartitionedDistributions.factorize_indices(A, [1, 3, 6, 5, 8, 10]) === nothing
            @test PartitionedDistributions.factorize_indices(A, [2, 4, 6]) === nothing
            inds = LinearIndices(A) .<= 10
            @test PartitionedDistributions.factorize_indices(A, inds) == ([1, 2, 3, 4, 5], [1, 2], [1])
            inds = mod.(LinearIndices(A) .+ 3, 5) .> 2
            @test PartitionedDistributions.factorize_indices(A, inds) == ([1, 5], [1, 2, 3, 4], [1, 2, 3])
        end
    end

    @testset "_reshape" begin
        udist = Normal()
        @test PartitionedDistributions._reshape(udist, ()) === udist
        @test PartitionedDistributions._reshape(udist, (1,)) === reshape(udist, (1,))
        @test PartitionedDistributions._reshape(udist, (1, 1)) === reshape(udist, (1, 1))

        mvdist = MvNormal(randn(12), rand_pdmat(PDMat{Float64}, 12))
        @test PartitionedDistributions._reshape(mvdist, (3, 4)) === reshape(mvdist, (3, 4))
        @test PartitionedDistributions._reshape(mvdist, (3, 4, 1)) === reshape(mvdist, (3, 4, 1))

        rdist = reshape(mvdist, (3, 4))
        rdist2 = PartitionedDistributions._reshape(rdist, (2, 6))
        @test rdist2 isa Distributions.ReshapedDistribution
        @test rdist2 === reshape(mvdist, (2, 6))

        rudist = reshape(udist, ())
        @test rudist isa Distributions.ReshapedDistribution{0}
        @test PartitionedDistributions._reshape(rudist, ()) === udist

        mv1dist = MvNormal(randn(1), rand_pdmat(PDMat{Float64}, 1))
        rudist2 = reshape(mv1dist, ())
        @test rudist2 isa Distributions.ReshapedDistribution{0}
        @test PartitionedDistributions._reshape(rudist2, ()) === rudist2
        @test PartitionedDistributions._reshape(rudist2, (1, 1)) === reshape(mv1dist, (1, 1))
    end

    @testset "_logbesseli / _logbesselk" begin
        logbesseli = PartitionedDistributions._logbesseli
        logbesselk = PartitionedDistributions._logbesselk
        ts = exp10.(range(-3, 3; length = 25))
        @testset "small orders match the scaled library functions" begin
            @testset for ν in (0.0, 0.5, 1.0, 2.5, 9.5), t in ts
                @test logbesseli(ν, t) ≈ log(besselix(ν, t)) + t
                @test logbesselk(ν, t) ≈ log(besselkx(ν, t)) - t
            end
            @testset "negative orders (I only)" begin
                @testset for ν in (-0.5,), t in ts
                    @test logbesseli(ν, t) ≈ log(besselix(ν, t)) + t
                end
            end
        end
        @testset "large orders match exact recurrences" begin
            # includes regimes where besselix underflows and besselkx overflows
            @testset for n in (10, 25, 60, 150, 500, 1000), t in (0.05, 1.0, 30.0, 900.0)
                @test logbesseli(n, t) ≈ logbesseli_recurrence(n, t) rtol = 1.0e-6
                @test logbesselk(n, t) ≈ logbesselk_recurrence(n, t) rtol = 1.0e-6
            end
        end
        @testset "smooth in the order across the library/asymptotic switch" begin
            # the switch happens where the scaled library value would leave (-690, 690);
            # a jump there would show up in the third differences over ν, which are otherwise
            # of order h³ / ν² ≈ 3e-6
            t = 0.01
            νs = 74:0.25:86
            lki = [logbesselk(ν, t) for ν in νs]
            lii = [logbesseli(ν, t) for ν in νs]
            @test minimum(lki .+ t) < 690 < maximum(lki .+ t)
            @test minimum(lii .- t) < -690 < maximum(lii .- t)
            @test all(diff(lki) .> 0)
            @test all(diff(lii) .< 0)
            @test maximum(abs, diff(diff(diff(lki)))) < 1.0e-5
            @test maximum(abs, diff(diff(diff(lii)))) < 1.0e-5
        end
        @testset "argument types" begin
            @test logbesseli(2.5f0, 3.0f0) isa Float32
            @test logbesselk(2.5f0, 3.0f0) isa Float32
            @test logbesseli(2, 3) isa Float64
            @test logbesselk(200.0f0, 1.0f0) ≈ logbesselk_recurrence(200, 1.0)
            @test logbesseli(200.0f0, 1.0f0) ≈ logbesseli_recurrence(200, 1.0)
        end
    end
end
