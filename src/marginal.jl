"""
    marginal(dist, keep_indices...)

Return the marginal distribution of `dist` at the indices `keep_indices`.

`keep_indices` should index into the support of `dist`.

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

function marginal(dist::Distributions.Distribution{Distributions.ArrayLikeVariate{N}}, _i1, _inds...) where {N}
    ax = axes(dist)
    inds = to_indices(size(dist), ax, (_i1, _inds...))
    _validate_indices(inds)
    ninds = length(inds)
    return if ninds == 1
        i = inds[1]
        if i isa Base.Slice
            return vec(dist)
        elseif i isa AbstractArray && ndims(i) > 1
            lin_inds = @views LinearIndices(ax)[i]
            dist_marg = _marginal_impl(dist, vec(lin_inds))
            return reshape(dist_marg, size(i))
        else
            return _marginal_impl(dist, i)
        end
    elseif ninds >= N
        inds_head = inds[1:N]
        if all(Base.Fix2(isa, Base.Slice), inds_head)
            dist_marg = dist
        else
            dist_marg = _marginal_impl(dist, inds_head...)
        end
        if ninds == N
            return dist_marg
        else
            sz = @views size(LinearIndices(ax)[inds...])
            return reshape(dist_marg, sz)
        end
    else
        throw(ArgumentError("Incorrect number of indices for array-variate distribution"))
    end
end

function _marginal_impl(dist::Distributions.UnivariateDistribution, i)
    all(isone, i) || throw(ArgumentError("Too many indices for univariate distribution"))
    i isa Int && return dist
    return vec(dist)
end

function _marginal_impl(dist::Distributions.AbstractMvNormal, i)
    μ = Distributions.mean(dist)
    Σ = Distributions.cov(dist)
    μ_i = view(μ, i)
    if iszero(ndims(μ_i))
        return Distributions.Normal(μ_i[], sqrt(only(Σ[i, i])))
    else
        return Distributions.MvNormal(μ_i, _pdview(Σ, i))
    end
end
function _marginal_impl(dist::Distributions.MatrixNormal, i1, i2)
    M_i = view(dist.M, i1, i2)
    if iszero(ndims(M_i))
        return Distributions.Normal(M_i[], sqrt(dist.U[i1, i1] * dist.V[i2, i2]))
    else
        return Distributions.MatrixNormal(M_i, _pdview(dist.U, i1), _pdview(dist.V, i2))
    end
end
function _marginal_impl(dist::Distributions.MatrixNormal, i)
    return _marginal_impl(vec(dist), i)
end
function _marginal_impl(dist::Distributions.MvLogNormal, i)
    dist_marg_norm = marginal(dist.normal, i)
    if dist_marg_norm isa Distributions.UnivariateDistribution
        return Distributions.LogNormal(dist_marg_norm.μ, dist_marg_norm.σ)
    else # dist_marg_norm isa Distributions.AbstractMvNormal
        return Distributions.MvLogNormal(dist_marg_norm)
    end
end
function _marginal_impl(dist::Distributions.GenericMvTDist, i)
    μ_i = view(dist.μ, i)
    if iszero(ndims(μ_i))
        return muladd(Distributions.TDist(dist.df), sqrt(only(dist.Σ[i, i])), μ_i[])
    else
        return Distributions.GenericMvTDist(dist.df, μ_i, _pdview(dist.Σ, i))
    end
end
function _marginal_impl(dist::Distributions.MixtureModel, i)
    return Distributions.MixtureModel(
        marginal.(Distributions.components(dist), Ref(i)),
        Distributions.probs(dist),
    )
end
function _marginal_impl(dist::Distributions.MixtureModel{Distributions.ArrayLikeVariate{N}}, i1, i2, irest...) where {N}
    inds = (i1, i2, irest...)
    length(inds) == N || throw(ArgumentError("Too many indices for array-variate distribution"))
    return Distributions.MixtureModel(
        marginal.(Distributions.components(dist), Ref.(inds)...),
        Distributions.probs(dist),
    )
end
if isdefined(Distributions, :JointOrderStatistics)
    function _marginal_impl(dist::Distributions.JointOrderStatistics, i)
        ranks = view(dist.ranks, i)
        if iszero(ndims(ranks))
            return Distributions.OrderStatistic(dist.dist, dist.n, ranks[])
        else
            return Distributions.JointOrderStatistics(dist.dist, dist.n, ranks)
        end
    end
end
if isdefined(Distributions, :ProductDistribution)
    function _marginal_impl(
            dist::Distributions.ProductDistribution{N, M},
            i1, i2, irest...,
        ) where {N, M}
        inds = (i1, i2, irest...)
        length(inds) == N || throw(ArgumentError("Incorrect number of indices for array-variate distribution"))
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
    function _marginal_impl(
            dist::Distributions.ProductDistribution{N, M},
            lin_i,
        ) where {N, M}
        ax = axes(dist)
        cart = CartesianIndices(ax)[lin_i]
        batch_inds = map(c -> CartesianIndex(ntuple(k -> c[M + k], Val(N - M))), cart)
        within_inds = map(c -> CartesianIndex(ntuple(k -> c[k], Val(M))), cart)
        unique_batches = unique(batch_inds)
        groups = map(b -> within_inds[batch_inds .== b], unique_batches)
        allequal(length.(groups)) || throw(
            ArgumentError(
                "linear indices select different numbers of elements from different components; " *
                    "cannot represent result as a ProductDistribution"
            )
        )
        marg_dists = map(unique_batches, groups) do b, ws
            comp = dist.dists[b]
            M == 0 && return comp
            return marginal(comp, LinearIndices(axes(comp))[ws])
        end
        return Distributions.ProductDistribution(vec(marg_dists))
    end
end
if isdefined(Distributions, :Product)
    function _marginal_impl(dist::Distributions.Product, i)
        marginals = @views dist.v[i]
        if marginals isa Distributions.Distribution
            return marginals
        else
            return Distributions.Product(marginals)
        end
    end
end
