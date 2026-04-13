"""
    pointwise_conditional_logpdfs(dist, x) -> logpdfs

Compute pointwise conditional logpdf of `x` for a given distribution.

Returns a collection with the same structure as `x` where each scalar is the
log-PDF of the distribution conditioned on all other elements of `x` and evaluated
only on the corresponding element of `x`.

Concretely, for a multivariate distribution with PDF ``p(x | θ)`` with indices
``i`` and parameters `θ``, this computes ``\\log p(x_i | x_{-i}, θ)`` for all ``i`` in
`\\text{LinearIndices}(x)`, where ``x_{-i}=`` `x[Not(i)]`. The returned
collection has the same shape as `x`. For array-variate distributions, this is
equivalent to `logpdf(conditional(dist, x, i), x[i]) for i in LinearIndices(x)]`
but is generally much more efficient.

See also: [`pointwise_conditional_logpdfs!!`](@ref), [`conditional`](@ref)

# Examples

Here's an example with a multivariate normal distribution:

```jldoctest pointwise_conditional_logpdfs
julia> using Distributions, PartitionedDistributions

julia> dist = MvNormal([ 0.8, -0.9], [1.3  0.7;  0.7 0.5]);

julia> x = [2.9, 0.4];

julia> pointwise_conditional_logpdfs(dist, x)
2-element Vector{Float64}:
 -0.47172139161049054
  0.012188177057073202
```

Here's an example with a NamedTuple-variate distribution:

```jldoctest pointwise_conditional_logpdfs
julia> nt_dist = product_distribution((x = dist, y = Normal())); # NamedTuple-variate distribution

julia> z = (; x, y=0.7)
(x = [2.9, 0.4], y = 0.7)

julia> pointwise_conditional_logpdfs(nt_dist, z)
(x = [-0.47172139161049054, 0.012188177057073202], y = -1.1639385332046728)
```
"""
function pointwise_conditional_logpdfs(dist::Distributions.Distribution, x)
    logp = _similar_logpdf(dist, x)
    return pointwise_conditional_logpdfs!!(logp, dist, x)
end

"""
    pointwise_conditional_logpdfs!!(logpdfs, dist, x) -> logpdfs

Maybe-in-place version of [`pointwise_conditional_logpdfs`](@ref).

If all scalar values in `logpdfs` can be mutated, then `logpdfs`
is filled in-place and returned. Otherwise, a new collection is returned.
"""
pointwise_conditional_logpdfs!!

function _logpdf_eltype(dist::Distributions.Distribution, x)
    return typeof(log(one(promote_type(eltype(x), Distributions.partype(dist)))))
end

# inefficient fallback for array-variate distributions
function pointwise_conditional_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.Distribution{Distributions.ArrayLikeVariate{N}},
        x::AbstractArray{<:Number, N},
    ) where {N}
    map!(logp, eachindex(x)) do i
        return Distributions.logpdf(conditional(dist, x, i), x[i])
    end
    return logp
end

function pointwise_conditional_logpdfs!!(::Number, dist::Distributions.UnivariateDistribution, x::Number)
    return Distributions.logpdf(dist, x)
end

# Array-variate normal distribution
function pointwise_conditional_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.AbstractMvNormal,
        x::AbstractVector{<:Number},
    )
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    λ = _pd_diag_inv(Σ)
    g = Σ \ (x - μ)
    return @. logp = (log(λ) - g^2 / λ - log2π) / 2
end
function pointwise_conditional_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvNormalCanon,
        x::AbstractVector{<:Number},
    )
    (; h, J) = dist
    λ = LinearAlgebra.diag(J)
    cov_inv_x = _pdmul(J, x)
    return @. logp = (log(λ) - (cov_inv_x - h)^2 / λ - log2π) / 2
end
function pointwise_conditional_logpdfs!!(
        logp::AbstractMatrix{<:Number},
        dist::Distributions.MatrixNormal,
        x::AbstractMatrix{<:Number},
    )
    (; M, U, V) = dist
    λU = _pd_diag_inv(U)
    λV = _pd_diag_inv(V)
    g = U \ (x - M) / V
    return @. logp = (log(λU) + log(λV') - g^2 / λU / λV' - log2π) / 2
end

# Multivariate log-normal distribution
function pointwise_conditional_logpdfs!!(
        logp::AbstractVector{<:Number},
        dist::Distributions.MvLogNormal,
        x::AbstractVector{<:Number},
    )
    logx = log.(x)
    pointwise_conditional_logpdfs!!(logp, dist.normal, logx)
    logp .-= logx
    return logp
end

# Array-variate t-distribution
function pointwise_conditional_logpdfs!!(
        logp::AbstractVector{T},
        dist::Distributions.GenericMvTDist,
        x::AbstractVector{<:Number},
    ) where {T <: Number}
    (; μ, Σ) = dist
    ν = dist.df
    νi = ν + length(dist) - 1
    α = (νi + 1) / 2
    logc = SpecialFunctions.loggamma(α) - SpecialFunctions.loggamma(νi / 2) - T(logπ) / 2
    λ = _pd_diag_inv(Σ)
    d = x - μ
    g = Σ \ d
    sqmahal = LinearAlgebra.dot(d, g)
    return map!(logp, λ, g) do λi, gi
        γ = gi^2 / λi
        β = ν + sqmahal - γ
        return logc - α * log1p(γ / β) + (log(λi) - log(β)) / 2
    end
end

# Mixtures of multivariate distributions
# NOTE: rand and logpdf for mixture fails on matrix-variate and higher-dimensional distributions
function pointwise_conditional_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.AbstractMixtureModel{Distributions.ArrayLikeVariate{N}},
        x::AbstractArray{<:Number, N}
    ) where {N}
    logp_k = similar(logp)
    fill!(logp, -Inf)
    logp_x = first(logp)

    K = Distributions.ncomponents(dist)
    for (k, w_k) in zip(1:K, Distributions.probs(dist))
        dist_k = Distributions.component(dist, k)
        logp_x_k = log(w_k) + Distributions.logpdf(dist_k, x)
        logp_x = LogExpFunctions.logaddexp(logp_x, logp_x_k)
        pointwise_conditional_logpdfs!!(logp_k, dist_k, x)
        logp .= LogExpFunctions.logaddexp.(logp, logp_x_k .- logp_k)
    end

    logp .= logp_x .- logp

    return logp
end

# work around type instability in partype(::AbstractMixtureModel)
# https://github.com/JuliaStats/Distributions.jl/blob/3d304c26f1cffd6a5bcd24fac2318be92877f4d5/src/mixtures/mixturemodel.jl#L170C41-L170C48
function _logpdf_eltype(dist::Distributions.AbstractMixtureModel, x::AbstractArray)
    prob_type = eltype(Distributions.probs(dist))
    components = Distributions.components(dist)
    component_type = if isconcretetype(eltype(components))  # all components are the same type
        _logpdf_eltype(first(components), x)
    else
        mapreduce(Base.Fix2(_logpdf_eltype, x), promote_type, components)
    end
    return promote_type(component_type, typeof(log(oneunit(prob_type))))
end

if isdefined(Distributions, :JointOrderStatistics)
    function pointwise_conditional_logpdfs!!(
            logp::AbstractVector{T},
            dist::Distributions.JointOrderStatistics,
            x::AbstractVector{<:Number},
        ) where {T <: Number}
        (; n, ranks) = dist
        m = length(x)

        if m == 1
            logp[begin] = Distributions.logpdf(dist, x)
            return logp
        end

        x_ext = Iterators.flatten((x, last(x)))
        ranks_ext = Iterators.flatten((ranks, n + 1))

        udist = dist.dist
        xi = first(x)
        ri = si = first(ranks)
        loggi = SpecialFunctions.loggamma(T(si))
        logdi = Distributions.logcdf(udist, xi)
        for (i, (xi_plus, ri_plus)) in enumerate(Iterators.drop(zip(x_ext, ranks_ext), 1))
            si_plus = ri_plus - ri
            si_gap = si + si_plus
            logdi_plus = if i == m
                Distributions.logccdf(udist, xi_plus)
            else
                Distributions.logdiffcdf(udist, xi_plus, xi)
            end
            logdi_gap = LogExpFunctions.logaddexp(logdi, logdi_plus)

            loggi_plus = SpecialFunctions.loggamma(T(si_plus))
            loggi_gap = SpecialFunctions.loggamma(T(si_gap))
            log_beta = loggi + loggi_plus - loggi_gap

            logpi = Distributions.logpdf(udist, xi)

            # log-pdf is basically a change-of-variables times a ratio of Dirichlets,
            # where all terms cancel except for the ones that change depending on whether
            # ranks[i] is observed or not.
            logp[i] =
                logpi + (si - 1) * logdi + (si_plus - 1) * logdi_plus -
                (si_gap - 1) * logdi_gap - log_beta

            (xi, ri, si, logdi, loggi) = (xi_plus, ri_plus, si_plus, logdi_plus, loggi_plus)
        end
        return logp
    end
end

# Product of array-variate distributions
if isdefined(Distributions, :ProductDistribution)
    function pointwise_conditional_logpdfs!!(
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
                pointwise_conditional_logpdfs!!(logp_i, dist_i, x_i)
            end
        end
        return logp
    end
end
if isdefined(Distributions, :Product)
    function pointwise_conditional_logpdfs!!(
            logp::AbstractVector{<:Number},
            dist::Distributions.Product,
            x::AbstractVector{<:Number},
        )
        logp .= Distributions.logpdf.(dist.v, x)
        return logp
    end
end
if isdefined(Distributions, :ProductNamedTupleDistribution)
    function _similar_logpdf(
            dist::Distributions.ProductNamedTupleDistribution, x::NamedTuple{K}
        ) where {K}
        return map(_similar_logpdf, NamedTuple{K}(dist.dists), x)
    end
    function pointwise_conditional_logpdfs(
            dist::Distributions.ProductNamedTupleDistribution,
            x::NamedTuple,
        )
        logp = _similar_logpdf(dist, x)
        return pointwise_conditional_logpdfs!!(logp, dist, x)
    end

    function pointwise_conditional_logpdfs!!(
            logp::NamedTuple{K},
            dist::Distributions.ProductNamedTupleDistribution,
            x::NamedTuple,
        ) where {K}
        return map(
            pointwise_conditional_logpdfs!!,
            logp,
            NamedTuple{K}(dist.dists),
            NamedTuple{K}(x),
        )
    end
end

function pointwise_conditional_logpdfs!!(
        logp::AbstractArray{<:Number, N},
        dist::Distributions.ReshapedDistribution{N},
        x::AbstractArray{<:Number, N},
    ) where {N}
    x_reshape = reshape(x, size(dist.dist))
    logp_reshape = reshape(logp, size(dist.dist))
    pointwise_conditional_logpdfs!!(logp_reshape, dist.dist, x_reshape)
    return logp
end

# Helper functions

function _pd_diag_inv(A::PDMats.AbstractPDMat)
    T = typeof(float(oneunit(eltype(A))))
    I = LinearAlgebra.Diagonal(ones(T, axes(A, 1)))
    return PDMats.invquad(A, I)
end

# hack to aboid ambiguity with *(::AbstractPDMat, ::DimArray)
_pdmul(A::PDMats.AbstractPDMat, b::StridedVector) = A * b
function _pdmul(A::PDMats.AbstractPDMat, b::AbstractVector)
    T = Base.promote_eltype(A, b)
    y = similar(b, T)
    LinearAlgebra.mul!(y, A, b)
    return y
end

function _similar_logpdf(dist::Distributions.UnivariateDistribution, x::Number)
    return zero(_logpdf_eltype(dist, x))
end
function _similar_logpdf(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate}, x
    )
    return similar(x, _logpdf_eltype(dist, x))
end
