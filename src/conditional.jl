"""
    conditional(dist, x, keep_indices...)

Given an array-variate distribution `dist` and a point `x` in its support, return the
distribution of `x[keep_indices...]` conditioned on the remaining elements of `x`.

`keep_indices` should index into the support of `dist`. The number of indices must
match the number of dimensions of the support: one index for multivariate distributions,
or one index per dimension for array-variate distributions. Linear indexing (a single
index for a multi-dimensional support) is not supported.

See also: [`marginal`](@ref)

# Examples

By providing a single index, we can compute the univariate distribution of a
single element of a multivariate normal distribution conditioned on the other
element(s).

```jldoctest conditional
julia> using Distributions, InvertedIndices, PartitionedDistributions

julia> dist = MvNormal([1.0, 2.0], [1.0 0.5; 0.5 1.0]);

julia> conditional(dist, [1.0, 2.0], 1)       # specify index to keep
Normal{Float64}(μ=1.0, σ=0.8660254037844386)

julia> conditional(dist, [1.0, 2.0], 1:1)     # specify indices to keep
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

function conditional(dist::Distributions.AbstractMvNormal, x::AbstractVector, _i)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    i = to_indices(x, (_i,))[1]
    ic = Not(i)
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    Σ_cond, B, _ = _schur_complement_and_factor(Σ, i)
    δ = @views x[ic] .- μ[ic]
    μ_cond = @views muladd(B', δ, μ[i])
    if iszero(ndims(μ_cond))
        return Distributions.Normal(μ_cond[], sqrt(Σ_cond))
    else
        return Distributions.MvNormal(μ_cond, Σ_cond)
    end
end
function conditional(dist::Distributions.MvNormalCanon, x::AbstractVector, i)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    ic = Not(i)
    J = Distributions.invcov(dist)
    J_cond = view(J, i, i)
    J_ic_i = view(J, ic, i)
    h_cond = @views dist.h[i] .- (J_ic_i' * x[ic])
    if iszero(ndims(h_cond))
        return Distributions.NormalCanon(h_cond[], J_cond[])
    else
        return Distributions.MvNormalCanon(h_cond, LinearAlgebra.Symmetric(J_cond))
    end
end
function conditional(dist::Distributions.MatrixNormal, x::AbstractMatrix, inds::Vararg{Any, 2})
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    # also Theorem 2.3.12 of Gupta & Nagar (2000) "Matrix variate distributions" https://doi.org/10.1201/9780203749289
    i1, i2 = to_indices(x, inds)
    ic1 = Not(i1)
    ic2 = Not(i2)
    M = Distributions.mean(dist)
    Ucond, BU, _ = _schur_complement_and_factor(dist.U, i1)
    Vcond, BV, _ = _schur_complement_and_factor(dist.V, i2)
    dX12 = @views x[ic1, i2] - M[ic1, i2]   # observed rows, kept cols
    dX21 = @views x[i1, ic2] - M[i1, ic2]   # kept rows, observed cols
    dX11 = @views x[ic1, ic2] - M[ic1, ic2] # observed rows, observed cols
    Mcond = view(M, i1, i2) + BU' * dX12 + (dX21 - BU' * dX11) * BV
    return Distributions.MatrixNormal(Mcond, Ucond, Vcond)
end
function conditional(dist::Distributions.MvLogNormal, x::AbstractVector, i)
    dist_cond_norm = conditional(dist.normal, log.(x), i)
    if dist_cond_norm isa Distributions.UnivariateDistribution
        return Distributions.LogNormal(Distributions.mean(dist_cond_norm), Distributions.std(dist_cond_norm))
    else # dist_cond_norm isa Distributions.AbstractMvNormal
        return Distributions.MvLogNormal(_mvnormal(dist_cond_norm))
    end
end
function conditional(dist::Distributions.GenericMvTDist, x::AbstractVector, i)
    # https://en.wikipedia.org/wiki/Multivariate_t-distribution#Conditional_Distribution
    (; μ, Σ) = dist
    ν = dist.df
    ic = Not(i)
    Σ_cond, B, Σ_ic = _schur_complement_and_factor(Σ, i)
    δ = @views x[ic] .- μ[ic]
    d = PDMats.invquad(Σ_ic, δ)
    μ_cond = muladd(B', δ, view(μ, i))
    ν_cond = ν + length(δ)

    if iszero(ndims(μ_cond))
        σ_cond = sqrt(Σ_cond * (ν + d) / ν_cond)
        return Distributions.TDist(ν_cond) * σ_cond + μ_cond[]
    else
        # avoid recomputing a Cholesky decomposition
        Σ_cond *= (ν + d) / ν_cond
        return Distributions.GenericMvTDist(ν_cond, μ_cond, Σ_cond)
    end
end
if isdefined(Distributions, :ProductDistribution)
    function conditional(
            dist::Distributions.ProductDistribution{N, M},
            x::AbstractArray{<:Real, N},
            inds::Vararg{Any, N},
        ) where {N, M}
        ind_in_component = inds[1:M]
        ind_component = inds[(M + 1):N]
        selected_dists = view(dist.dists, ind_component...)
        if iszero(ndims(selected_dists))
            selected_dist = selected_dists[]
            M == 0 && return selected_dist
            x_comp = view(x, ntuple(_ -> Colon(), Val(M))..., ind_component...)
            return conditional(selected_dist, x_comp, ind_in_component...)
        elseif M == 0
            # Scalar components are independent: conditioning on other components has no effect
            return Distributions.product_distribution(selected_dists)
        else
            x_selected = view(x, ntuple(_ -> Colon(), Val(M))..., ind_component...)
            batch_dims = ntuple(k -> M + k, ndims(selected_dists))
            cond_dists = map(
                (d, x_d) -> conditional(d, x_d, ind_in_component...),
                selected_dists,
                eachslice(x_selected; dims = batch_dims),
            )
            return Distributions.product_distribution(cond_dists)
        end
    end
end
if isdefined(Distributions, :Product)
    conditional(dist::Distributions.Product, ::AbstractVector, i) = dist.v[i]
end
