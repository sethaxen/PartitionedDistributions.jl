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

function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.AbstractMvNormal,
        x::AbstractVector{<:Number},
    )
    μ = Distributions.mean(dist)
    v = Distributions.var(dist)
    @. logp = Distributions.logpdf(Distributions.Normal(μ, sqrt(v); check_args = false), x)
    return logp
end
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvNormalCanon,
        x::AbstractVector{<:Number},
    )
    (; μ, J) = dist
    v = _pd_diag_inv(J)
    @. logp = Distributions.logpdf(Distributions.Normal(μ, sqrt(v); check_args = false), x)
    return logp
end
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{<:Number},
        dist::Distributions.MatrixNormal,
        x::AbstractMatrix{<:Number},
    )
    (; M, U, V) = dist
    vU = LinearAlgebra.diag(U)
    vV = LinearAlgebra.diag(V)
    logp .= Distributions.logpdf.(
        Distributions.Normal.(M, sqrt.(vU) .* sqrt.(vV'); check_args = false),
        x,
    )
    return logp
end

function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{T},
        dist::Distributions.GenericMvTDist,
        x::AbstractVector{<:Number},
    ) where {T <: Number}
    (; μ, Σ) = dist
    ν = T(dist.df)
    α = T(ν + 1) / 2
    logc = _tdist_lognorm(ν)
    v = LinearAlgebra.diag(Σ)
    return @. logp = logc - α * log1p(((x - μ) / sqrt(v))^2 / ν) - log(v) / 2
end

function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.MatrixTDist,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; M, Σ, Ω) = dist
    ν = T(dist.ν)
    α = T(ν + 1) / 2
    logc = _tdist_lognorm(ν) + log(ν) / 2
    vΣ = LinearAlgebra.diag(Σ)
    vΩ = LinearAlgebra.diag(Ω)
    return @. logp = logc - α * log1p(((x - M) / (sqrt(vΣ) * sqrt(vΩ')))^2) - (log(vΣ) + log(vΩ')) / 2
end

function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.Dirichlet,
        x::AbstractVector{<:Number},
    )
    (; alpha, alpha0) = dist
    @. logp = Distributions.logpdf(Distributions.Beta(alpha, alpha0 - alpha; check_args = false), x)
    return logp
end

function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.Multinomial,
        x::AbstractVector{<:Number},
    )
    (; n, p) = dist
    @. logp = Distributions.logpdf(Distributions.Binomial(n, p; check_args = false), x)
    return logp
end

function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.DirichletMultinomial,
        x::AbstractVector{<:Number},
    )
    (; n, α, α0) = dist
    @. logp = Distributions.logpdf(Distributions.BetaBinomial(n, α, α0 - α; check_args = false), x)
    return logp
end

# LKJ distribution: off-diagonal marginals are 2 * Beta(a, a) - 1 with a = η - 1 + d / 2
# (§3.3 of Lewandowski, Kurowicka & Joe (2009), doi: 10.1016/j.jmva.2009.04.008);
# diagonal entries are identically 1.
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.LKJ,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; d, η) = dist
    a = η - 1 + d / 2
    lognorm = -(2a - 1) * logtwo - SpecialFunctions.logbeta(a, a)
    broadcast!(logp, 1:d, (1:d)', x) do i, j, r
        if i == j
            return isone(r) ? zero(T) : T(-Inf)
        else
            return _lkj_offdiag_logpdf(a, lognorm, r)
        end
    end
    return logp
end
# log-density at r of 2 * Beta(a, a) - 1, with the normalization constant hoisted into lognorm
function _lkj_offdiag_logpdf(a, lognorm, r)
    abs(r) <= 1 || return oftype(lognorm, -Inf)
    return lognorm + LogExpFunctions.xlogy(a - 1, (1 - r) * (1 + r))
end

# Wishart distribution: diagonal entries are scaled χ² (Gamma) distributed, and off-diagonal
# entries follow a variance-gamma distribution, see
# https://en.wikipedia.org/wiki/Wishart_distribution#Marginal_distribution_of_matrix_elements
function pointwise_marginal_logpdfs!!(
        logp::AbstractMatrix{T},
        dist::Distributions.Wishart,
        x::AbstractMatrix{<:Number},
    ) where {T <: Number}
    (; S) = dist
    df = T(dist.df)
    ν = (df - 1) / 2
    lognorm = -SpecialFunctions.loggamma(df / 2) - ν * logtwo - T(logπ) / 2
    v = LinearAlgebra.diag(S)
    d = size(S, 1)
    broadcast!(logp, 1:d, (1:d)', x, S, v, v') do i, j, xij, Sij, vi, vj
        if i == j
            gamma = Distributions.Gamma(df / 2, 2 * vi; check_args = false)
            return Distributions.logpdf(gamma, xij)
        else
            return _wishart_offdiag_logpdf(ν, lognorm, Sij, vi, vj, xij)
        end
    end
    return logp
end
function _wishart_offdiag_logpdf(ν, lognorm, Sij, vi, vj, x)
    a = sqrt(vi) * sqrt(vj)
    loga = (log(vi) + log(vj)) / 2
    ρ = Sij / a
    ρ2c = (1 - ρ) * (1 + ρ)  # 1 - ρ²
    b = a * ρ2c
    return lognorm + _logbesselk_times_power(ν, abs(x) / b) + ρ * x / b - loga + (ν - 1 // 2) * log(ρ2c)
end

# von Mises–Fisher distribution
# The marginal density of dot(v, x) for a von Mises–Fisher variate x
# is given in Eq. 25 of:
#   Romanazzi, M. (2014). "Discriminant Analysis with High Dimensional von Mises–Fisher
#   Distributions." Athens Journal of Sciences, 1(4), 225–240.
#   http://www.atiner.gr/journals/sciences/2014-1-4-1-Romanazzi.pdf
# For the D = 3 case, see also Eq 4.4 of:
#   Mardia, K. V. & Edwards, R. (1982). "Weighted distributions and rotating caps."
#   Biometrika, 69(2), 323–330.
#   https://doi.org/10.1093/biomet/69.2.323
function pointwise_marginal_logpdfs!!(
        logp::AbstractVector{T},
        dist::Distributions.VonMisesFisher,
        x::AbstractVector{<:Number},
    ) where {T <: Number}
    (; μ, κ) = dist
    D = length(μ)
    Dlower = D - 1
    ν = T(D // 2 - 1)
    νlower = T(Dlower // 2 - 1)
    # log C_D(κ) + (D - 1) / 2 * log(2π), recomputed with `_logbesseli` rather than taken from
    # `dist.logCκ`, which overflows for large D with small κ
    logc = ν * log(κ) - T(log2π) / 2 - _logbesseli(ν, κ)
    logp .= _vmf_marginal_logpdf.(logc, κ, νlower, μ, x)
    return logp
end
function _vmf_marginal_logpdf(logc, κ, ν, μi, xi)
    abs(xi) <= 1 || return oftype(logc, -Inf)
    xi2c = (1 - xi) * (1 + xi)  # 1 - xᵢ², accurate near |xᵢ| = 1
    μi2c = max(zero(μi), (1 - μi) * (1 + μi))  # guard against rounding of the unit vector μ
    s = κ * sqrt(μi2c) * sqrt(xi2c)
    # (1 - xᵢ²)^ν, avoiding 0 * -Inf for D == 3 at |xᵢ| == 1
    logjac = LogExpFunctions.xlogy(ν, xi2c)
    return logc + κ * μi * xi + logjac + _logbesseli_over_power(ν, s)
end

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
function _tdist_lognorm(ν)
    return (
        SpecialFunctions.loggamma((ν + 1) / 2) -
        SpecialFunctions.loggamma(ν / 2) -
        (log(ν) + logπ) / 2
    )
end

# log(t^ν K_ν(t)), which is finite as t → 0 for ν > 0 since K_ν(t) ~ Γ(ν) 2^(ν - 1) t^(-ν)
function _logbesselk_times_power(ν, t)
    iszero(t) && return SpecialFunctions.loggamma(ν) + (ν - 1) * oftype(float(t), logtwo)
    return ν * log(t) + _logbesselk(ν, t)
end

# log(I_ν(s) / s^ν), which is finite as s → 0
function _logbesseli_over_power(ν, s)
    iszero(s) && return -ν * oftype(float(s), logtwo) - SpecialFunctions.loggamma(ν + 1)
    return _logbesseli(ν, s) - ν * log(s)
end
