"""
    marginal(dist, keep_indices...)

Return the marginal distribution of `dist` at the indices `keep_indices`.

`keep_indices` should index into the support of `dist`. The number of indices must
match the number of dimensions of the support: one index for multivariate distributions,
or one index per dimension for array-variate distributions. Linear indexing (a single
index for a multi-dimensional support) is not supported.

See also: [`conditional`](@ref)

# Examples

```jldoctest marginal
julia> using Distributions, InvertedIndices, PartitionedDistributions

julia> dist = MvNormal([1.0, 2.0, 3.0], [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]);

julia> marginal(dist, 1)       # specify index to keep
Normal{Float64}(μ=1.0, σ=1.0)

julia> marginal(dist, [1, 3])  # specify indices to keep
MvNormal{Float64, PDMats.PDMat{Float64, Matrix{Float64}}, SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}}(
dim: 2
μ: [1.0, 3.0]
Σ: [1.0 0.25; 0.25 1.0]
)

julia> marginal(dist, Not(2))  # alternatively, specify index to keep
MvNormal{Float64, PDMats.PDMat{Float64, Matrix{Float64}}, SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}}(
dim: 2
μ: [1.0, 3.0]
Σ: [1.0 0.25; 0.25 1.0]
)
```
"""
marginal

function marginal(dist::Distributions.AbstractMvNormal, i)
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    μ_i = view(μ, i)
    if iszero(ndims(μ_i))
        return Distributions.Normal(μ_i[], sqrt(only(Σ[i, i])))
    else
        return Distributions.MvNormal(μ_i, _pdview(Σ, i))
    end
end
function marginal(dist::Distributions.MatrixNormal, i1, i2)
    M_i = view(dist.M, i1, i2)
    if iszero(ndims(M_i))
        return Distributions.Normal(M_i[], sqrt(dist.U[i1, i1] * dist.V[i2, i2]))
    else
        return Distributions.MatrixNormal(M_i, _pdview(dist.U, i1), _pdview(dist.V, i2))
    end
end
function marginal(dist::Distributions.MvLogNormal, i)
    dist_marg_norm = marginal(dist.normal, i)
    if dist_marg_norm isa Distributions.UnivariateDistribution
        return Distributions.LogNormal(dist_marg_norm.μ, dist_marg_norm.σ)
    else # dist_marg_norm isa Distributions.AbstractMvNormal
        return Distributions.MvLogNormal(dist_marg_norm)
    end
end
function marginal(dist::Distributions.GenericMvTDist, i)
    μ_i = view(dist.μ, i)
    if iszero(ndims(μ_i))
        return muladd(Distributions.TDist(dist.df), sqrt(only(dist.Σ[i, i])), μ_i[])
    else
        return Distributions.GenericMvTDist(dist.df, μ_i, _pdview(dist.Σ, i))
    end
end
function marginal(dist::Distributions.MixtureModel, inds...)
    return Distributions.MixtureModel(
        marginal.(Distributions.components(dist), Ref.(inds)...),
        Distributions.probs(dist),
    )
end
if isdefined(Distributions, :JointOrderStatistics)
    function marginal(dist::Distributions.JointOrderStatistics, i)
        ranks = view(dist.ranks, i)
        if iszero(ndims(ranks))
            return Distributions.OrderStatistic(dist.dist, dist.n, ranks[])
        else
            return Distributions.JointOrderStatistics(dist.dist, dist.n, ranks)
        end
    end
end
if isdefined(Distributions, :ProductDistribution)
    function marginal(
            dist::Distributions.ProductDistribution{N, M},
            inds::Vararg{Any, N},
        ) where {N, M}
        ind_in_component = inds[1:M]
        ind_component = inds[(M + 1):N]
        selection = @views dist.dists[ind_component...]
        if selection isa Distributions.Distribution
            M == 0 && return selection
            # recurse into within-component marginal
            return marginal(selection, ind_in_component...)
        elseif M == 0
            return Distributions.product_distribution(selection)
        else
            marg_dists = map(d -> marginal(d, ind_in_component...), selection)
            return Distributions.product_distribution(marg_dists)
        end
    end
end
if isdefined(Distributions, :Product)
    function marginal(dist::Distributions.Product, i)
        marginals = @views dist.v[i]
        if marginals isa Distributions.Distribution
            return marginals
        else
            return Distributions.Product(marginals)
        end
    end
end
