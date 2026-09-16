"""
    pointwise_marginal_logpdfs(dist, x) -> logp

Compute pointwise marginal log-PDF of `x` for a given distribution.

Returns a collection with the same structure as `x` where each scalar is the
log-PDF of the marginal distribution of the corresponding element of `x`, evaluated
at that element.

For array-variate distributions, this is equivalent to

```julia
[logpdf(marginal(dist, i), x[i]) for i in LinearIndices(x)]
```
but is generally much more efficient.

See [`pointwise_marginal_logpdfs!!`](@ref) for a maybe-in-place version.

See also: [`marginal`](@ref), [`pointwise_conditional_logpdfs`](@ref)

# Examples

Here's an example with a multivariate normal distribution:

```jldoctest pointwise_marginal_logpdfs
julia> using Distributions, PartitionedDistributions

julia> dist = MvNormal([ 0.8, -0.9], [1.3  0.7;  0.7 0.5]);

julia> x = [2.9, 0.4];

julia> pointwise_marginal_logpdfs(dist, x)
2-element Vector{Float64}:
 -2.7462745115922638
 -2.2623649429247
```

Here's an example with a `NamedTuple`-variate distribution:

```jldoctest pointwise_marginal_logpdfs
julia> nt_dist = product_distribution((x = dist, y = Normal()));

julia> z = (; x, y=0.7)
(x = [2.9, 0.4], y = 0.7)

julia> pointwise_marginal_logpdfs(nt_dist, z)
(x = [-2.7462745115922638, -2.2623649429247], y = -1.1639385332046728)
```
"""
function pointwise_marginal_logpdfs(dist::Distributions.Distribution, x)
    logp = _similar_logpdf(dist, x)
    return pointwise_marginal_logpdfs!!(logp, dist, x)
end

"""
    pointwise_marginal_logpdfs!!(logp, dist, x) -> logpdfs

Maybe-in-place version of [`pointwise_marginal_logpdfs`](@ref).

If all scalar values in `logp` can be mutated, then `logp`
is filled in-place and returned. Otherwise, a new collection is returned.
"""
pointwise_marginal_logpdfs!!

# inefficient fallback for array-variate distributions
function pointwise_marginal_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.Distribution{Distributions.ArrayLikeVariate{N}},
        x::AbstractArray{<:Number, N},
    ) where {N}
    map!(logp, eachindex(x)) do i
        return Distributions.logpdf(marginal(dist, i), x[i])
    end
    return logp
end

function pointwise_marginal_logpdfs!!(::Number, dist::Distributions.UnivariateDistribution, x::Number)
    return Distributions.logpdf(dist, x)
end

# Array-variate normal distributions: elementwise marginals are univariate normals
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.AbstractMvNormal,
        x::AbstractVector{<:Number},
    )
    μ = Distributions.mean(dist)
    σ = sqrt.(Distributions.var(dist))
    logp .= Distributions.logpdf.(Distributions.Normal.(μ, σ; check_args = false), x)
    return logp
end
# avoid forming the full covariance matrix
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvNormalCanon,
        x::AbstractVector{<:Number},
    )
    σ = sqrt.(_pd_diag_inv(dist.J))
    logp .= Distributions.logpdf.(Distributions.Normal.(dist.μ, σ; check_args = false), x)
    return logp
end
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{<:Number},
        dist::Distributions.MatrixNormal,
        x::AbstractMatrix{<:Number},
    )
    (; M, U, V) = dist
    σU = sqrt.(LinearAlgebra.diag(U))
    σV = sqrt.(LinearAlgebra.diag(V))
    logp .= Distributions.logpdf.(Distributions.Normal.(M, σU .* σV'; check_args = false), x)
    return logp
end

# Multivariate t-distribution: elementwise marginals are affine univariate t-distributions.
# The normalization constant of the t-distribution (two loggamma evaluations) is hoisted out
# of the elementwise broadcast.
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{T},
        dist::Distributions.GenericMvTDist,
        x::AbstractVector{<:Number},
    ) where {T <: Number}
    (; μ, Σ) = dist
    ν = dist.df
    α = (ν + 1) / 2
    logc = _tdist_lognorm(T, ν)
    σ = sqrt.(LinearAlgebra.diag(Σ))
    return @. logp = logc - α * log1p(((x - μ) / σ)^2 / ν) - log(σ)
end

# Matrix-variate t-distribution
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.MatrixTDist,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; ν, M, Σ, Ω) = dist
    α = (ν + 1) / 2
    logc = _tdist_lognorm(T, ν)
    σΣ = sqrt.(LinearAlgebra.diag(Σ))
    σΩ = sqrt.(LinearAlgebra.diag(Ω)) ./ sqrt(ν)
    return @. logp = logc - α * log1p(((x - M) / (σΣ * σΩ'))^2 / ν) - log(σΣ * σΩ')
end

# Dirichlet distribution: elementwise marginals are Beta distributions
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.Dirichlet,
        x::AbstractVector{<:Number},
    )
    (; alpha, alpha0) = dist
    logp .= Distributions.logpdf.(Distributions.Beta.(alpha, alpha0 .- alpha; check_args = false), x)
    return logp
end

# Multinomial distribution: elementwise marginals are Binomial distributions
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.Multinomial,
        x::AbstractVector{<:Number},
    )
    (; n, p) = dist
    logp .= Distributions.logpdf.(Distributions.Binomial.(n, p; check_args = false), x)
    return logp
end

# Dirichlet-multinomial distribution: elementwise marginals are BetaBinomial distributions
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.DirichletMultinomial,
        x::AbstractVector{<:Number},
    )
    (; n, α, α0) = dist
    logp .= Distributions.logpdf.(Distributions.BetaBinomial.(n, α, α0 .- α; check_args = false), x)
    return logp
end

# LKJ distribution: off-diagonal marginals are 2 * Beta(a, a) - 1 with a = η - 1 + d / 2
# (Lewandowski, Kurowicka & Joe (2009), doi: 10.1016/j.jmva.2009.04.008);
# diagonal entries are identically 1.
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.LKJ,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; d, η) = dist
    a = η - 1 + d / 2
    beta = Distributions.Beta(a, a; check_args = false)
    for j in axes(x, 2), i in axes(x, 1)
        r = x[i, j]
        logp[i, j] = if i == j
            isone(r) ? zero(T) : T(-Inf)
        else
            Distributions.logpdf(beta, (r + 1) / 2) - T(logtwo)
        end
    end
    return logp
end

# Wishart distribution: diagonal entries are scaled χ² (Gamma) distributed, and off-diagonal
# entries follow a variance-gamma distribution, see
# https://en.wikipedia.org/wiki/Wishart_distribution#Marginal_distribution_of_matrix_elements
# The log-density of Xᵢⱼ is written in terms of a = sqrt(Sᵢᵢ Sⱼⱼ) and det₂ = Sᵢᵢ Sⱼⱼ - Sᵢⱼ².
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.Wishart,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; df, S) = dist
    ν = (df - 1) / 2
    logc = -SpecialFunctions.loggamma(df / 2) - ν * T(logtwo) - T(logπ) / 2
    for j in axes(x, 2), i in axes(x, 1)
        xij = x[i, j]
        Sii, Sjj = S[i, i], S[j, j]
        if i == j
            logp[i, j] = Distributions.logpdf(Distributions.Gamma(df / 2, 2 * Sii; check_args = false), xij)
            continue
        end
        Sij = S[i, j]
        a = sqrt(Sii) * sqrt(Sjj)
        det2 = (a - Sij) * (a + Sij)
        logp[i, j] = if iszero(xij)
            # limit x → 0 of the density below, using K_ν(t) ~ Γ(ν) 2^(ν - 1) t^(-ν)
            logc + SpecialFunctions.loggamma(ν) + (ν - 1) * T(logtwo) + (ν - 1 // 2) * log(det2) - 2ν * log(a)
        else
            t = abs(xij) * a / det2
            logc + ν * log(abs(xij) / a) + _logbesselk(ν, t) + Sij * xij / det2 - log(det2) / 2
        end
    end
    return logp
end

# von Mises–Fisher distribution: the marginal density of the coordinate xᵢ of x ∈ S^(D-1) is
#   C_D(κ) (2π)^((D-1)/2) (1 - xᵢ²)^ν exp(κ μᵢ xᵢ) I_ν(s) / s^ν,   ν = (D - 3) / 2,
# with s = κ sqrt((1 - μᵢ²)(1 - xᵢ²)), obtained by integrating the density over the
# (D-2)-sphere of directions orthogonal to eᵢ. The normalizing constant C_D(κ) is recomputed with
# `_logbesseli` rather than taken from `dist` so that large D with small κ does not overflow.
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{T},
        dist::Distributions.VonMisesFisher,
        x::AbstractVector{<:Number},
    ) where {T <: Number}
    (; μ, κ) = dist
    D = length(μ)
    ν = (D - 3) / 2
    # log C_D(κ) + (D - 1) / 2 * log(2π)
    logc = (D / 2 - 1) * log(κ) - T(log2π) / 2 - _logbesseli(D / 2 - 1, κ)
    logp .= _vmf_marginal_logpdf.(logc, κ, ν, μ, x)
    return logp
end
function _vmf_marginal_logpdf(logc, κ, ν, μi, xi)
    abs(xi) <= 1 || return oftype(logc, -Inf)
    xi2c = (1 - xi) * (1 + xi)  # 1 - xᵢ², accurate near |xᵢ| = 1
    μi2c = max(zero(μi), (1 - μi) * (1 + μi))  # guard against rounding of the unit vector μ
    s = κ * sqrt(μi2c * xi2c)
    # (1 - xᵢ²)^ν, avoiding 0 * -Inf for D == 3 at |xᵢ| == 1
    logjac = iszero(ν) ? zero(logc) : ν * log(xi2c)
    return logc + κ * μi * xi + logjac + _logbesseli_over_power(ν, s)
end

# Mixtures of array-variate distributions: the marginal of a mixture is the mixture of the
# marginals of its components with the same weights.
function pointwise_marginal_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.AbstractMixtureModel{Distributions.ArrayLikeVariate{N}},
        x::AbstractArray{<:Number, N},
    ) where {N}
    logp_k = similar(logp)
    fill!(logp, -Inf)
    K = Distributions.ncomponents(dist)
    for (k, w_k) in zip(1:K, Distributions.probs(dist))
        dist_k = Distributions.component(dist, k)
        pointwise_marginal_logpdfs!!(logp_k, dist_k, x)
        logp .= LogExpFunctions.logaddexp.(logp, log(w_k) .+ logp_k)
    end
    return logp
end

# Product distributions: components are independent, so marginalize within each component
function pointwise_marginal_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.ProductDistribution{N, M},
        x::AbstractArray{<:Number, N},
    ) where {N, M}
    if M == 0
        logp .= Distributions.logpdf.(dist.dists, x)
    else
        dims = ntuple(i -> i + M, Val(N - M))  # product dimensions
        for (x_i, logp_i, dist_i) in
            zip(eachslice(x; dims), eachslice(logp; dims), dist.dists)
            pointwise_marginal_logpdfs!!(logp_i, dist_i, x_i)
        end
    end
    return logp
end

# NamedTuple-variate product distributions
function pointwise_marginal_logpdfs!!(
        logp::NamedTuple{K},
        dist::Distributions.ProductNamedTupleDistribution,
        x::NamedTuple,
    ) where {K}
    return map(
        pointwise_marginal_logpdfs!!,
        logp,
        NamedTuple{K}(dist.dists),
        NamedTuple{K}(x),
    )
end

# Reshaped distributions, just delegate to the underlying distribution and reshape
function pointwise_marginal_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.ReshapedDistribution{N},
        x::AbstractArray{<:Number, N},
    ) where {N}
    x_reshape = reshape(x, size(dist.dist))
    logp_reshape = reshape(logp, size(dist.dist))
    pointwise_marginal_logpdfs!!(logp_reshape, dist.dist, x_reshape)
    return logp
end

# Helper functions

# log normalization constant of the standard t-distribution with ν degrees of freedom
function _tdist_lognorm(::Type{T}, ν) where {T}
    return SpecialFunctions.loggamma((ν + 1) / 2) - SpecialFunctions.loggamma(ν / 2) - (log(ν) + T(logπ)) / 2
end

# log(I_ν(s) / s^ν), which is finite as s → 0
function _logbesseli_over_power(ν, s)
    iszero(s) && return -ν * oftype(float(s), logtwo) - SpecialFunctions.loggamma(ν + 1)
    return _logbesseli(ν, s) - ν * log(s)
end
