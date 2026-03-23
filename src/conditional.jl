"""
    conditional(dist, y, keep_indices...) -> ContinuousUnivariateDistribution

Compute a conditional univariate distribution.

Given an array-variate distribution `dist` and an array `y` in its support,
return the univariate distribution of `y[keep_indices...]` given
the remaining elements of `y`.

See also: [`marginal`](@ref)

# Examples

By providing a single index, we can compute the univariate distribution of a
single element of a multivariate normal distribution conditioned on the other
element(s).

```jldoctest conditional
julia> using Distributions, InvertedIndices, PartitionedDistributions

julia> dist = MvNormal([1.0, 2.0], [1.0 0.5; 0.5 1.0]);

julia> conditional(dist, [1.0, 2.0], 1)  # specify index to keep
Normal{Float64}(μ=1.0, σ=0.8660254037844386)

julia> conditional(dist, [1.0, 2.0], 1:2)  # specify indices to keep
FullNormal(
dim: 1
μ: [1.0]
Σ: [0.75;;]
)

julia> conditional(dist, [1.0, 2.0], Not(2))  # alternatively, specify index to condition on
FullNormal(
dim: 1
μ: [1.0]
Σ: [0.75;;]
)
```
"""
conditional

function conditional(dist::Distributions.AbstractMvNormal, y::AbstractVector, i)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    ic = Not(i)
    Σ_ic_i = @views Σ[ic, i]
    Σ_ic = @views Σ[ic, ic]
    inv_Σ_ic_Σ_ic_i = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ_ic)) \ Σ_ic_i
    Σ_cond = Σ[i, i] - inv_Σ_ic_Σ_ic_i' * Σ_ic_i  # Schur complement
    μ_cond = μ[i] + inv_Σ_ic_Σ_ic_i' * @views(y[ic] - μ[ic])
    dist_cond = if iszero(ndims(μ_cond))
        return Distributions.Normal(μ_cond[], sqrt(Σ_cond[]))
    else
        return Distributions.MvNormal(μ_cond, LinearAlgebra.Symmetric(Σ_cond))
    end
    # TODO: support trailing dimensions
    return dist_cond
end
function conditional(dist::Distributions.MvNormalCanon, y::AbstractVector, i)
    # TODO: handle this directly
    return conditional(Distributions.MvNormal(Distributions.mean(dist), Distributions.cov(dist)), y, i)
end
function conditional(dist::Distributions.MatrixNormal, y::AbstractMatrix, i...)
    vec_i = @view LinearIndices(y)[i...]
    if ndims(vec_i) < 2
        vec_y = vec(y)
        vec_dist = vec(dist)
        return conditional(vec_dist, vec_y, vec_i)
    else  # it's a MatrixNormal
        # TODO: work out how to compute this efficiently
        return conditional(dist, y, i)
    end
end
function conditional(dist::Distributions.MvLogNormal, y::AbstractVector, i)
    dist_cond_norm = conditional(dist.normal, log.(y), i)
    if dist_cond_norm isa Distributions.UnivariateDistribution
        return Distributions.LogNormal(dist_cond_norm.μ, dist_cond_norm.σ)
    else # dist_cond_norm isa Distributions.AbstractMvNormal
        # TODO: support trailing dimensions
        return Distributions.MvLogNormal(dist_cond_norm)
    end
end
function conditional(
    dist::Distributions.GenericMvTDist, y::AbstractVector, i,
)
    # https://en.wikipedia.org/wiki/Multivariate_t-distribution#Conditional_Distribution
    (; μ, Σ) = dist
    ν = dist.df
    ic = Not(i)
    Σ_ic_i = @view Σ[ic, i]
    Σ_ic = @view Σ[ic, ic]
    chol_Σ_ic = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ_ic))
    δ = @views y[ic] - μ[ic]
    d = LinearAlgebra.dot(δ, chol_Σ_ic \ δ)
    inv_Σ_ic_Σ_ic_i = chol_Σ_ic \ Σ_ic_i
    Σ_cond = @views Σ[i, i] - inv_Σ_ic_Σ_ic_i' * Σ_ic_i  # Schur complement
    μ_cond = @views μ[i] + inv_Σ_ic_Σ_ic_i' * δ
    ν_cond = ν + length(ic)

    if iszero(ndims(μ_cond))
        σ_cond = sqrt(Σ_cond * (ν + d) / ν_cond)
        return Distributions.TDist(ν_cond) * σ_cond + μ_cond[]
    else
        Σ_cond *= (ν + d) / ν_cond
        return Distributions.GenericMvTDist(ν_cond, μ_cond, LinearAlgebra.Symmetric(Σ_cond))
    end
end
if isdefined(Distributions, :ProductDistribution)
    # TODO: fix this implementation
    function conditional(
        dist::Distributions.ProductDistribution{N,M},
        y::AbstractArray{<:Real,N},
        i...,
    ) where {N,M}
        inds = Tuple(i)
        ind_in_component = inds[1:M]
        ind_component = inds[(M + 1):N]
        dist_i = dist.dists[inds[(M + 1):N]...]
        M == 0 && return dist_i
        y_i = y[fill(Colon(), M)..., ind_component...]
        ind_in_component = if length(ind_in_component) == 1
            ind_in_component[1]
        else
            CartesianIndex(ind_in_component)
        end
        return conditional(dist_i, y_i, ind_in_component)
    end
    function conditional(
        dist::Distributions.ProductDistribution{1,0}, ::AbstractVector, i...
    )
        dists_i = dist.dists[i...]
        if iszero(ndims(dists_i))
            return dists_i[]
        else
            return Distributions.product_distribution(dists_i)
        end
    end
end
if isdefined(Distributions, :Product)
    function conditional(dist::Distributions.Product, ::AbstractVector, i)
        return dist.v[i]
    end
end
