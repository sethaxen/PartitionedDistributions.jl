"""
    marginal(dist, keep_indices...)

Return the marginal distribution of `dist` at the indices `keep_indices`.

`keep_indices` should be valid indices for any element in the support of `dist`.

See also: [`conditional`](@ref)

# Examples

```jldoctest marginal
julia> using Distributions, InvertedIndices, PartitionedDistributions

julia> dist = MvNormal([1.0, 2.0, 3.0], [1.0 0.5 0.25; 0.5 1.0 0.5; 0.25 0.5 1.0]);

julia> marginal(dist, 1)  # specify index to keep
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
    μ_i = @views μ[i]
    Σ_i = @views Σ[i, i]
    dist_marg = if iszero(ndims(μ_i))
        return Distributions.Normal(μ_i[], sqrt(Σ_i[]))
    else
        return Distributions.MvNormal(μ_i, LinearAlgebra.Symmetric(Σ_i))
    end
    # TODO: support trailing dimensions
    return dist_marg
end
function marginal(dist::Distributions.MvNormalCanon, i)
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    μ_i = @views μ[i]
    Σ_i = @views Σ[i, i]
    dist_marg = if iszero(ndims(μ_i))
        λ = inv(Σ_i[])
        η = λ * μ_i[]
        return Distributions.NormalCanon(η, λ)
    else
        J_i = LinearAlgebra.Symmetric(inv(LinearAlgebra.Symmetric(Σ_i)) + I * eps(eltype(μ)))
        h_i = J_i * μ_i
        return Distributions.MvNormalCanon(h_i, J_i)
    end
    # TODO: support trailing dimensions
    return dist_marg
end
function marginal(dist::Distributions.MvLogNormal, i)
    dist_marg_norm = marginal(dist.normal, i)
    if dist_marg_norm isa Distributions.UnivariateDistribution
        return Distributions.LogNormal(dist_marg_norm.μ, dist_marg_norm.σ)
    else # dist_marg_norm isa Distributions.AbstractMvNormal
        # TODO: support trailing dimensions
        return Distributions.MvLogNormal(dist_marg_norm)
    end
end
function marginal(dist::Distributions.GenericMvTDist, i)
    μ_i = @views dist.μ[i]
    Σ_i = @views dist.Σ[i, i]
    dist_marg = if iszero(ndims(μ_i))
        return muladd(Distributions.TDist(dist.df), sqrt(Σ_i[]), μ_i[])
    else
        return Distributions.GenericMvTDist(dist.df, μ_i, LinearAlgebra.PDMat(LinearAlgebra.Symmetric(Σ_i)))
    end
    # TODO: support trailing dimensions
    return dist_marg
end
function marginal(dist::Distributions.MixtureModel, i)
    return Distributions.MixtureModel(
        marginal.(Distributions.components(dist), Ref(i)),
        Distributions.probs(dist),
    )
end
if isdefined(Distributions, :JointOrderStatistics)
    function marginal(dist::Distributions.JointOrderStatistics, i)
        ranks = @view dist.ranks[i]
        if iszero(ndims(ranks))
            return Distributions.OrderStatistic(dist.dist, dist.n, ranks[])
        else
            return Distributions.JointOrderStatistics(dist.dist, dist.n, ranks)
        end
    end
end
if isdefined(Distributions, :Product)
    function marginal(dist::Distributions.Product, i)
        return Distributions.Product(dist.v[i])
    end
end
