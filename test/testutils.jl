# Indexing test helpers: dimension-agnostic pieces (`default_axis_specs`, `default_base_index_tuple`,
# `example_multidim_linear_index_matrix`, trailing singletons) plus `ArrayLikeVariate{1}`-specific
# `example_vector_indices` overloads. For a new `N`-D distribution, prefer
# `test_axis_aligned_partition_combos(dist, y, default_axis_specs(dist))` when axis-aligned `Not`
# partitions make sense; otherwise supply custom per-axis vectors of length `N`.
using Distributions
using InvertedIndices: Not
using LinearAlgebra
using PDMats: PDMat, PDiagMat, ScalMat
using PartitionedDistributions
using Random
using Test

rand_pdmat(::Type{T}, n::Int) where {T} = rand_pdmat(Random.default_rng(), T, n)
function rand_pdmat(rng::AbstractRNG, ::Type{Matrix{S}}, n) where {S <: AbstractFloat}
    return Matrix(rand_pdmat(rng, PDMat{S}, n))
end
function rand_pdmat(rng::AbstractRNG, ::Type{<:PDMat{S}}, n) where {S <: AbstractFloat}
    A = randn(rng, S, n, n)
    return PDMat(Symmetric(A * A' + cbrt(eps(S)) * I))
end
function rand_pdmat(rng::AbstractRNG, ::Type{<:PDiagMat{S}}, n) where {S <: AbstractFloat}
    return PDiagMat(abs2.(randn(rng, S, n)))
end
function rand_pdmat(rng::AbstractRNG, ::Type{<:ScalMat{S}}, n) where {S <: AbstractFloat}
    return ScalMat(n, randn(rng, S)^2)
end

"""
    complement_linear(x, i) -> Not(...)

Complement of the selected linear indices of `x` for use with `marginal`/`conditional`
decomposition tests (same role as `Not(i)` when `i` selects along `axes(x, 1)`).
"""
function complement_linear(x::AbstractArray, i)
    li = LinearIndices(axes(x))
    sel = li[i]
    idxs = sel isa Integer ? [Int(sel)] : vec(sel)
    return Not(idxs)
end

function default_rtol(dist::Distributions.Distribution{<:ArrayLikeVariate}, atol::Real)
    rtol = cbrt(eps(float(eltype(dist))))
    return atol > 0 ? zero(rtol) : rtol
end

function default_rtol(dist::Distributions.ProductNamedTupleDistribution, atol::Real)
    return maximum(Base.Fix2(default_rtol, atol), dist.dists)
end

_isapprox(a, b; kwargs...) = isapprox(a, b; kwargs...)
function _isapprox(a::NamedTuple{K}, b::NamedTuple{K}; kwargs...) where {K}
    return all(zip(a, b)) do (a_i, b_i)
        return _isapprox(a_i, b_i; kwargs...)
    end
end

"""
    test_logpdf_decomposition(dist, x, inds, comp_inds)

Test the chain rule identity:

    logpdf(dist, x) ≈ logpdf(conditional(dist, x, inds...), x[inds...]) +
                      logpdf(marginal(dist, comp_inds...), x[comp_inds...])
"""
function test_logpdf_decomposition(dist, x, inds, comp_inds; atol::Real = 0, rtol::Real = default_rtol(dist, atol))
    cond_dist = conditional(dist, x, inds...)
    marg_dist = marginal(dist, comp_inds...)
    return @test logpdf(dist, x) ≈ logpdf(cond_dist, x[inds...]) + logpdf(marg_dist, x[comp_inds...]) rtol = rtol atol = atol
end

"""
    test_marginal_moments_match(dist, inds...; test_var::Bool=true, test_cov::Bool=false)

Test that moments of `marginal(dist, inds...)` match slices of the moments of `dist`.
"""
function test_marginal_moments_match(
        dist,
        inds...;
        test_var::Bool = true,
        test_cov::Bool = false,
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    return @testset "Marginal moments match" begin
        marg_dist = marginal(dist, inds...)
        @testset "Mean matches" begin
            mean_dist = mean(dist)
            mean_marg = mean(marg_dist)
            @test _isapprox(mean_marg, mean_dist[inds...]; rtol = rtol, atol = atol)
        end
        test_var && @testset "Variance matches" begin
            @test _isapprox(var(marg_dist), var(dist)[inds...]; rtol = rtol, atol = atol)
        end
        test_cov && @testset "Covariance matches" begin
            lin_inds = vec(LinearIndices(axes(dist))[inds...])
            @test cov(marg_dist) ≈ cov(dist)[lin_inds, lin_inds] rtol = rtol atol = atol
        end
    end
end

# --- Example index baskets (single linear-style argument; N == 1 semantics) ---

function _default_example_vector_indices(ax)
    n = length(ax)
    fi, la = first(ax), last(ax)
    mi = (fi + la) ÷ 2
    inds = Any[fi]
    n > 1 && push!(inds, la)
    n >= 2 && push!(inds, fi:min(fi + 1, la))
    n >= 3 && push!(inds, (la - 1):-1:fi)
    n >= 2 && push!(inds, [fi, la])
    n >= 3 && push!(inds, [fi, la, mi])
    push!(inds, Not(la))
    push!(inds, Bool[mod1(k, 2) == 1 for k in 1:n])
    push!(inds, Colon())
    return inds
end

function _example_vector_indices_scalmat_t(ax)
    n = length(ax)
    fi, la = first(ax), last(ax)
    inds = Any[fi]
    n >= 2 && push!(inds, fi:min(fi + 1, la))
    n >= 2 && push!(inds, [fi, la])
    push!(inds, Not(la))
    push!(inds, Bool[mod1(k, 2) == 1 for k in 1:n])
    return inds
end

# `:` omitted for ProductDistribution{1,0}: `marginal(dist, Not(:))` uses the linear-index
# path and errors with an empty selection (Product handles `Not(:)` differently).
function _example_vector_indices_productdistribution_scalar_len5(ax)
    return Any[
        1,
        5,
        1:3,
        [1, 3, 5],
        Not(1),
        Not(2:4),
        Bool[true, false, true, false, true],
    ]
end

function _example_vector_indices_product_scalar_len5(ax)
    return Any[_example_vector_indices_productdistribution_scalar_len5(ax)..., Colon()]
end

"""
    example_vector_indices(dist) -> Vector

Single-index-argument examples (linear / logical / `:` along the sole axis) for
`Distribution{ArrayLikeVariate{1}}`.
"""
function example_vector_indices(dist::Distributions.Distribution{Distributions.ArrayLikeVariate{1}})
    return _default_example_vector_indices(first(axes(dist)))
end
function example_vector_indices(dist::Distributions.ProductDistribution{1, 0})
    ax = first(axes(dist))
    return length(ax) == 5 ? _example_vector_indices_productdistribution_scalar_len5(ax) : _default_example_vector_indices(ax)
end
function example_vector_indices(dist::Distributions.Product)
    ax = first(axes(dist))
    return length(ax) == 5 ? _example_vector_indices_product_scalar_len5(ax) : _default_example_vector_indices(ax)
end
function example_vector_indices(dist::Distributions.GenericMvTDist)
    ax = first(axes(dist))
    return dist.Σ isa ScalMat ? _example_vector_indices_scalmat_t(ax) : _default_example_vector_indices(ax)
end

# --- Per-axis index lists (one selector per dimension; any `N ≥ 1`) ---

"""
    default_axis_specs(dist) -> NTuple{N, Vector{Any}}

For each dimension `d`, a small deterministic list of index objects to try in axis-aligned
partition tests (`test_axis_aligned_partition_combos`).
"""
function default_axis_specs(dist::Distributions.Distribution)
    return default_axis_specs(Tuple(map(Int, size(dist))))
end

function default_axis_specs(sz::NTuple{N, Int}) where {N}
    return ntuple(d -> _default_axis_spec_list(sz[d]), Val(N))
end

function _default_axis_spec_list(s::Int)
    v = Any[1:min(2, s)]
    push!(v, 1:1)
    s >= 2 && push!(v, [1, s])
    s >= 3 && push!(v, Not(s))
    push!(v, Bool[mod1(k, 2) == 1 for k in 1:s])
    return v
end

# --- Multidimensional array of linear indices (single argument; reshape path) ---

"""
    example_multidim_linear_index_matrix(dist) -> Union{Nothing,AbstractMatrix{Int}}

A 2D matrix of **linear** indices into `axes(dist)` with unique entries, such that
`marginal(dist, Im)` uses the `ndims(Im) > 1` branch. Returns `nothing` if `prod(size(dist)) < 2`.
"""
function example_multidim_linear_index_matrix(dist::Distributions.Distribution)
    sz = Tuple(map(Int, size(dist)))
    return example_multidim_linear_index_matrix(sz, LinearIndices(axes(dist)))
end

function example_multidim_linear_index_matrix(sz::NTuple{N, Int}, L::LinearIndices) where {N}
    prod(sz) < 2 && return nothing
    if N >= 2
        n1, n2 = min(2, sz[1]), min(2, sz[2])
        tail = ntuple(_ -> 1, Val(max(0, N - 2)))
        return [L[i, j, tail...] for i in 1:n1, j in 1:n2]
    else
        n = sz[1]
        n < 2 && return nothing
        if n == 2
            return reshape([1, 2], 2, 1)
        elseif n == 3
            return reshape([1, 2, 3], 3, 1)
        else
            return reshape(collect(1:4), 2, 2)
        end
    end
end

function test_multidim_linear_index_matrix_consistency(dist, y)
    Im = example_multidim_linear_index_matrix(dist)
    Im === nothing && return nothing
    return @testset "single-arg multidim array of linear indices (reshape path)" begin
        mvn = vec(dist)
        lin = vec(Im)
        x_sub = y[Im]
        marg_mat = marginal(dist, Im)
        cond_mat = conditional(dist, y, Im)
        @test isfinite(logpdf(marg_mat, x_sub))
        @test isfinite(logpdf(cond_mat, x_sub))
        # `vec(dist)` may exist but not support linear `marginal`/`conditional` (e.g. some
        # `ReshapedDistribution` wrappers); only compare when both calls succeed.
        try
            marg_vec = marginal(mvn, lin)
            @test logpdf(marg_mat, x_sub) ≈ logpdf(marg_vec, vec(x_sub))
            cond_vec = conditional(mvn, vec(y), lin)
            @test logpdf(cond_mat, x_sub) ≈ logpdf(cond_vec, vec(x_sub))
        catch e
            e isa MethodError || rethrow()
        end
    end
end

# --- Trailing singleton indices (any `N ≥ 0` leading indices) ---

const TRAILING_SINGLETON_SUFFIXES = ((1,), (1:1,), ([1],), (Colon(),))

function _test_trailing_singleton_core(dist, y, base::Tuple, trailers)
    ref_m = marginal(dist, base...)
    ref_x = y[base...]
    ref_logm = logpdf(ref_m, ref_x)
    ref_c = conditional(dist, y, base...)
    ref_logc = logpdf(ref_c, ref_x)
    return @testset for t in trailers
        full = (base..., t...)
        m = marginal(dist, full...)
        xv = y[full...]
        @test logpdf(m, xv) ≈ ref_logm
        @test logpdf(conditional(dist, y, full...), xv) ≈ ref_logc
    end
end

"""
    default_base_index_tuple(dist) -> Tuple

A non-empty leading index tuple with one range per dimension (for trailing-singleton checks).
"""
function default_base_index_tuple(dist::Distributions.Distribution)
    return default_base_index_tuple(Tuple(map(Int, size(dist))))
end

function default_base_index_tuple(sz::NTuple{N, Int}) where {N}
    N == 0 && return (1,)  # `ArrayLikeVariate{0}` still uses `1` in `marginal`/`conditional`
    return ntuple(d -> 1:min(2, sz[d]), Val(N))
end

"""
    test_trailing_singleton_indices(dist, y)

After `N` leading indices that fully address `ndims(dist)`, trailing `1`, `1:1`, `[1]`, and `:`
should follow `getindex`. Requires `y` to be an `AbstractArray` so `y[inds...]` supports
trailing singleton dimensions (e.g. `fill(0.5)` for univariate, not a bare `Float64`).
"""
function test_trailing_singleton_indices(dist, y)
    sz = size(dist)
    N = length(sz)
    base = default_base_index_tuple(sz)
    return @testset "trailing singleton indices" begin
        _test_trailing_singleton_core(dist, y, base, TRAILING_SINGLETON_SUFFIXES)
    end
end

"""
    test_axis_aligned_partition_combos(dist, y, axis_specs::NTuple{N, Vector{Any}}; kwargs...)

For each dimension `d` and each `id ∈ axis_specs[d]`, test logpdf decomposition with an
axis-aligned keep / complement partition. Then moment checks on the Cartesian product of
`axis_specs`, trailing singletons, and multidim linear-index reshape consistency.
"""
function test_axis_aligned_partition_combos(dist, y, axis_specs::NTuple{N, Vector{Any}}; kwargs...) where {N}
    @testset for d in 1:N
        @testset for id in axis_specs[d]
            keep = ntuple(k -> k == d ? id : Colon(), N)
            comp = ntuple(k -> k == d ? Not(id) : Colon(), N)
            test_logpdf_decomposition(dist, y, keep, comp; kwargs...)
            test_logpdf_decomposition(dist, y, comp, keep; kwargs...)
        end
    end
    @testset for combo in Iterators.product(axis_specs...)
        test_marginal_moments_match(dist, combo...; test_cov = true, kwargs...)
    end
    test_trailing_singleton_indices(dist, y)
    test_multidim_linear_index_matrix_consistency(dist, y)
    return nothing
end

"""
    test_all_index_combos(dist, y; kwargs...)

For `ArrayLikeVariate{1}`: decomposition + moments for `example_vector_indices`, then trailing
singletons and (when applicable) a multidim linear-index matrix in the single-index slot.
"""
function test_all_index_combos(dist, y; kwargs...)
    @testset for i in example_vector_indices(dist)
        test_logpdf_decomposition(dist, y, (i,), (Not(i),); kwargs...)
        test_logpdf_decomposition(dist, y, (Not(i),), (i,); kwargs...)
        test_marginal_moments_match(dist, i; kwargs...)
    end
    test_trailing_singleton_indices(dist, y)
    test_multidim_linear_index_matrix_consistency(dist, y)
    return nothing
end

"""
    test_univariate_arraylike_indexing(dist, x::AbstractArray{<:Real,0})

`conditional` requires `x` to be an `AbstractArray` matching `ArrayLikeVariate{0}`; trailing
singleton indices work on `x` but not on a bare scalar.
"""
function test_univariate_arraylike_indexing(dist, x::AbstractArray{<:Real, 0})
    @testset "marginal/conditional match joint" begin
        @test logpdf(marginal(dist, 1), x[]) ≈ logpdf(dist, x[])
        @test logpdf(conditional(dist, x, 1), x[]) ≈ logpdf(dist, x[])
    end
    test_trailing_singleton_indices(dist, x)
    @test_throws MethodError conditional(dist, x[], 1)
    return nothing
end
