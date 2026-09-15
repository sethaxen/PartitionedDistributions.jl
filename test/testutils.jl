using DimensionalData
using Distributions
using InvertedIndices: Not
using LinearAlgebra
using PDMats: PDMat, PDiagMat, ScalMat
using PartitionedDistributions
using Random
using SpecialFunctions: besselix, besselkx
using Test

"""
    wrap_array(T::Type{<:AbstractArray}, x::AbstractArray) -> T

Wrap `x` if necessary to convert to type `T`.
"""
wrap_array(::Type{T}, x::AbstractArray) where {T <: AbstractArray} = convert(T, x)
function wrap_array(::Type{T}, x::AbstractArray) where {T <: DimArray}
    d = ntuple(i -> (X, Y, Z)[i](axes(x, i)), ndims(x))
    return T(x, d)
end

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
    test_pointwise_matches_conditional(dist, x; atol, rtol)

For array-variate `dist`, check that `pointwise_conditional_logpdfs` agrees with
`logpdf(conditional(dist, x, i), x[i])` for each `i in eachindex(x)`.
Also checks `axes(logp) == axes(x)` so e.g. `DimensionalData.DimArray` inputs preserve dimensions on output.
For [`ProductNamedTupleDistribution`](@ref), recurse into each factor (independent blocks).

Does not apply to distributions without a working [`conditional`](@ref) (e.g. [`JointOrderStatistics`](@ref)).
"""
function test_pointwise_matches_conditional(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate},
        x::AbstractArray{<:Number};
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    logp = pointwise_conditional_logpdfs(dist, x)
    @test axes(logp) == axes(x)
    logp_ref = [logpdf(conditional(dist, x, i), x[i]) for i in LinearIndices(x)]
    @test logp ≈ logp_ref rtol = rtol atol = atol
    return nothing
end

function test_pointwise_matches_conditional(
        dist::Distributions.Distribution{<:Distributions.Univariate},
        x::Number;
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    logp = pointwise_conditional_logpdfs(dist, x)
    @test logp ≈ logpdf(dist, fill(x)) rtol = rtol atol = atol
    @test pointwise_conditional_logpdfs!!(oftype(x, NaN), dist, x) == logp
    return nothing
end

"""
    test_pointwise_marginal_matches_reference(dist, x, logp_ref; atol, rtol)

For array-variate `dist`, check that `pointwise_marginal_logpdfs(dist, x)` agrees with
`logp_ref`, that `axes(logp) == axes(x)` (so e.g. `DimensionalData.DimArray` inputs preserve
dimensions on output), and that `pointwise_marginal_logpdfs!!` fills its first argument in-place.
"""
function test_pointwise_marginal_matches_reference(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate},
        x::AbstractArray{<:Number},
        logp_ref;
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    logp = pointwise_marginal_logpdfs(dist, x)
    @test axes(logp) == axes(x)
    @test logp ≈ logp_ref rtol = rtol atol = atol
    logp2 = similar(logp)
    @test pointwise_marginal_logpdfs!!(logp2, dist, x) === logp2
    @test logp2 == logp
    return nothing
end

"""
    test_pointwise_marginal_matches_marginal(dist, x; atol, rtol)

For array-variate `dist`, check that `pointwise_marginal_logpdfs` agrees with
`logpdf(marginal(dist, i), x[i])` for each `i in eachindex(x)`
(see [`test_pointwise_marginal_matches_reference`](@ref)).

Does not apply to distributions without a working [`marginal`](@ref) (e.g. [`Dirichlet`](@ref)).
"""
function test_pointwise_marginal_matches_marginal(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate},
        x::AbstractArray{<:Number};
        kwargs...,
    )
    logp_ref = [logpdf(marginal(dist, i), x[i]) for i in LinearIndices(x)]
    return test_pointwise_marginal_matches_reference(dist, x, logp_ref; kwargs...)
end

function test_pointwise_marginal_matches_marginal(
        dist::Distributions.Distribution{<:Distributions.Univariate},
        x::Number;
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    logp = pointwise_marginal_logpdfs(dist, x)
    @test logp ≈ logpdf(dist, x) rtol = rtol atol = atol
    @test pointwise_marginal_logpdfs!!(oftype(logp, NaN), dist, x) == logp
    return nothing
end

# for distributions without a working `conditional`
function test_pointwise_matches_marginal(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate},
        x::AbstractArray{<:Number};
        atol::Real = 0,
        rtol::Real = default_rtol(dist, atol),
    )
    logp = pointwise_conditional_logpdfs(dist, x)
    @test axes(logp) == axes(x)
    lp = logpdf(dist, x)
    logp_ref = [lp - logpdf(marginal(dist, Not(i)), x[Not(i)]) for i in LinearIndices(x)]
    @test logp ≈ logp_ref rtol = rtol atol = atol
    return nothing
end


"""
    numerical_marginal_logpdf(dist, x, i; npts=100_000) -> Float64

Log-density of the marginal of the `i`th element of a 3-component `Dirichlet`, evaluated at
`x[i]`, computed by integrating the joint density over the remaining free coordinate with a
midpoint rule. Used as an implementation-independent reference.
"""
function numerical_marginal_logpdf(dist::Dirichlet, x::AbstractVector, i::Int; npts::Int = 100_000)
    length(dist) == 3 || throw(ArgumentError("only 3-component Dirichlet is supported"))
    j, k = filter(!=(i), 1:3)
    v = Float64(x[i])
    w = 1 - v
    h = w / npts
    y = zeros(3)
    y[i] = v
    lps = map(1:npts) do m
        t = (m - 0.5) * h
        y[j] = t
        y[k] = w - t
        return logpdf(dist, y)
    end
    lmax = maximum(lps)
    return lmax + log(sum(lp -> exp(lp - lmax), lps)) + log(h)
end

"""
    numerical_marginal_logpdf(dist::LKJ, x, i, j; npts=1_000) -> Float64

Log-density of the marginal of the off-diagonal entry `(i, j)` of a `3 × 3` `LKJ`, evaluated at
`x[i, j]`, computed by integrating the joint density over the other two free correlations with
a midpoint rule. Used as an implementation-independent reference.
"""
function numerical_marginal_logpdf(dist::LKJ, x::AbstractMatrix, i::Int, j::Int; npts::Int = 1_000)
    dist.d == 3 || throw(ArgumentError("only 3 × 3 LKJ is supported"))
    i != j || throw(ArgumentError("only off-diagonal entries have a density"))
    k = only(filter(∉((i, j)), 1:3))
    R = Matrix{Float64}(I, 3, 3)
    R[i, j] = R[j, i] = x[i, j]
    h = 2 / npts
    lps = Float64[]
    for a in 1:npts, b in 1:npts
        R[i, k] = R[k, i] = -1 + (a - 0.5) * h
        R[j, k] = R[k, j] = -1 + (b - 0.5) * h
        isposdef(R) || continue
        push!(lps, logpdf(dist, R))
    end
    lmax = maximum(lps)
    return lmax + log(sum(lp -> exp(lp - lmax), lps)) + 2 * log(h)
end

"""
    enumerated_marginal_logpdf(dist, x, i) -> Float64

Log-pmf of the marginal of the `i`th element of a 3-category `Multinomial` or
`DirichletMultinomial`, evaluated at `x[i]`, computed by summing the joint pmf over all
configurations of the other two counts. Used as an implementation-independent reference.
"""
function enumerated_marginal_logpdf(
        dist::Union{Multinomial, DirichletMultinomial}, x::AbstractVector, i::Int,
    )
    length(dist) == 3 || throw(ArgumentError("only 3-category distributions are supported"))
    j, k = filter(!=(i), 1:3)
    r = dist.n - x[i]
    y = zeros(Int, 3)
    y[i] = x[i]
    lps = map(0:r) do t
        y[j] = t
        y[k] = r - t
        return logpdf(dist, y)
    end
    lmax = maximum(lps)
    return lmax + log(sum(lp -> exp(lp - lmax), lps))
end

"""
    logbesselk_recurrence(n::Int, t) -> Float64

`log(besselk(n, t))` for integer order `n ≥ 0`, computed in log-space from `K_0` and `K_1` with
the (forward-stable) recurrence `K_{n+1}(t) = K_{n-1}(t) + (2n / t) K_n(t)`.
Used as an implementation-independent reference for large orders.
"""
function logbesselk_recurrence(n::Int, t)
    lk = log(besselkx(0.0, t)) - t
    n == 0 && return lk
    r = besselkx(1.0, t) / besselkx(0.0, t)  # K_1 / K_0
    for m in 1:n
        lk += log(r)  # log K_m
        m == n && break
        r = 1 / r + 2m / t  # K_{m+1} / K_m
    end
    return lk
end

"""
    logbesseli_recurrence(n::Int, t) -> Float64

`log(besseli(n, t))` for integer order `n ≥ 0`, computed in log-space from `I_0` with the
ratios `I_{m+1} / I_m` obtained by (backward-stable) Miller recurrence.
Used as an implementation-independent reference for large orders.
"""
function logbesseli_recurrence(n::Int, t; extra::Int = 200)
    nmax = n + extra + ceil(Int, t)
    r = 0.0  # I_{nmax+1} / I_{nmax} ≈ 0
    logratios = 0.0
    for m in nmax:-1:1
        r = 1 / (2m / t + r)  # I_m / I_{m-1}
        m <= n && (logratios += log(r))
    end
    return log(besselix(0.0, t)) + t + logratios
end

"""
    test_pointwise_marginal_mc_normalization(dist, nsamples; atol, quantile_pairs)

Monte Carlo check of the pointwise marginal densities that only requires `rand(dist)`.
For `x ~ dist` with marginal density `pᵢ` of `xᵢ` and any interval `[lo, hi]`,

    E[1{lo ≤ xᵢ ≤ hi} / pᵢ(xᵢ)] = hi - lo,

since the expectation is `∫_{lo}^{hi} pᵢ(t) / pᵢ(t) dt`. Draw `nsamples` samples, compute the
pointwise marginal log-densities, and for every element and every pair of sample quantiles in
`quantile_pairs` compare the log of the Monte Carlo estimate (via logsumexp) with `log(hi - lo)`.
Restricting to an interval within the bulk keeps the variance of the estimator finite even when
`pᵢ` vanishes at the edge of its support; using two different intervals also checks the shape
of `pᵢ`, not only its normalization.
"""
function test_pointwise_marginal_mc_normalization(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate},
        nsamples::Int;
        atol::Real = 0.02,
        quantile_pairs = ((0.25, 0.75), (0.005, 0.995)),
    )
    x1 = rand(dist)
    xs = Matrix{eltype(x1)}(undef, length(x1), nsamples)
    logps = Matrix{Float64}(undef, length(x1), nsamples)
    for n in 1:nsamples
        x = n == 1 ? x1 : rand(dist)
        xs[:, n] = vec(x)
        logps[:, n] = vec(pointwise_marginal_logpdfs(dist, x))
    end
    @test all(isfinite, logps)
    @testset for i in 1:length(x1), (ql, qh) in quantile_pairs
        xi = view(xs, i, :)
        lo, hi = quantile(xi, (ql, qh))
        v = [-logps[i, n] for n in 1:nsamples if lo <= xi[n] <= hi]
        vmax = maximum(v)
        logest = vmax + log(sum(exp, v .- vmax)) - log(nsamples)
        @test logest ≈ log(hi - lo) atol = atol
    end
    return nothing
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
