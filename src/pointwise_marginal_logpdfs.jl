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

function pointwise_marginal_logpdfs!!(::Number, dist::Distributions.UnivariateDistribution, x::Number)
    return Distributions.logpdf(dist, x)
end

# Array-variate normal distributions: elementwise marginals are univariate normals
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.AbstractMvNormal,
        x::AbstractVector{<:Number},
    )
    return _normal_logpdfs!(logp, Distributions.mean(dist), Distributions.var(dist), x)
end
# avoid forming the full covariance matrix
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvNormalCanon,
        x::AbstractVector{<:Number},
    )
    return _normal_logpdfs!(logp, dist.μ, _pd_diag_inv(dist.J), x)
end
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{<:Number},
        dist::Distributions.MatrixNormal,
        x::AbstractMatrix{<:Number},
    )
    (; M, U, V) = dist
    σ2 = LinearAlgebra.diag(U) .* LinearAlgebra.diag(V)'
    return _normal_logpdfs!(logp, M, σ2, x)
end

# Multivariate log-normal distribution
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvLogNormal,
        x::AbstractVector{<:Number},
    )
    logx = log.(x)
    pointwise_marginal_logpdfs!!(logp, dist.normal, logx)
    logp .-= logx
    return logp
end

# Multivariate t-distribution: elementwise marginals are affine univariate t-distributions
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.GenericMvTDist,
        x::AbstractVector{<:Number},
    )
    return _tdist_logpdfs!(logp, dist.df, dist.μ, LinearAlgebra.diag(dist.Σ), x)
end

# Matrix-variate t-distribution
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{<:Number},
        dist::Distributions.MatrixTDist,
        x::AbstractMatrix{<:Number},
    )
    (; ν, M, Σ, Ω) = dist
    σ2 = LinearAlgebra.diag(Σ) .* LinearAlgebra.diag(Ω)' ./ ν
    return _tdist_logpdfs!(logp, ν, M, σ2, x)
end

# Helper functions

# elementwise log-pdf of Normal(μ, sqrt(σ2)) at x
function _normal_logpdfs!(logp, μ, σ2, x)
    return @. logp = -(log(σ2) + (x - μ)^2 / σ2 + log2π) / 2
end

# elementwise log-pdf of μ + sqrt(σ2) * TDist(ν) at x
function _tdist_logpdfs!(logp::AbstractArray{T}, ν, μ, σ2, x) where {T}
    α = (ν + 1) / 2
    logc = SpecialFunctions.loggamma(α) - SpecialFunctions.loggamma(ν / 2) - (log(ν) + T(logπ)) / 2
    return @. logp = logc - α * log1p((x - μ)^2 / (ν * σ2)) - log(σ2) / 2
end
