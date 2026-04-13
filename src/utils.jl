# Compute
# - the Schur complement S = A[i,i] - A[ic,i]' * (A[ic,ic] \ A[ic,i]) as an AbstractPDMat or scalar
# - the factor B = A[ic,ic] \ A[ic,i]
# - the complement submatrix A[ic,ic] as an AbstractPDMat.
# Assumes i is an Int, a Not, or a Slice or vector selector
function _schur_complement_and_factor(A::AbstractMatrix, i)
    ic = Not(i)
    A_ic_i = view(A, ic, i)
    A_ic_ic = PDMats.AbstractPDMat(view(A, ic, ic))
    B = A_ic_ic \ A_ic_i
    i isa Int && return A[i, i] - A_ic_i' * B, B, A_ic_ic
    A_ii = view(A, i, i)
    return PDMats.PDMat(LinearAlgebra.Symmetric(A_ii - A_ic_i' * B)), B, A_ic_ic
end
# For diagonal pdmats, off-diagonal blocks are zero: B = 0, S = diagonal submatrix.
function _schur_complement_and_factor(A::Union{PDMats.PDiagMat, PDMats.ScalMat}, i)
    ic = Not(i)
    Σ_ic = _pdview(A, ic)
    n_ic = size(Σ_ic, 1)
    i isa Int && return A[i, i], FillArrays.Zeros(n_ic), Σ_ic
    n_i = size(A, 1) - n_ic
    return _pdview(A, i), FillArrays.Zeros(n_ic, n_i), Σ_ic
end

# symmetric submatrix view of an AbstractPDMat, assumes i is not an Int
_pdview(A::PDMats.AbstractPDMat, i) = PDMats.AbstractPDMat(view(A, i, i))
_pdview(A::PDMats.PDiagMat, i) = PDMats.PDiagMat(view(A.diag, i))
_pdview(A::PDMats.ScalMat, i) = PDMats.ScalMat(size(view(A, i, i), 1), first(A))

_mvnormal(dist::Distributions.MvNormal) = dist
_mvnormal(dist::Distributions.AbstractMvNormal) = Distributions.MvNormal(Distributions.mean(dist), Distributions.cov(dist))

function _validate_indices(inds)
    return foreach(_validate_index, inds)
end

_validate_index(i::Base.Slice) = nothing
_validate_index(i::Base.LogicalIndex) = nothing
_validate_index(i::Int) = nothing
function _validate_index(i::Base.AbstractArray)
    allunique(i) || throw(ArgumentError("Indices must be unique"))
    return nothing
end

"""
    factorize_indices(x, inds) -> inds_per_dim

Given a numeric `N`-dimensional array `x` and an array `inds` usable as a
single index into `x`, return `inds_per_dim`, a length `N` tuple of index vectors
such that
```julia
vec(x[inds_per_dim...]) == x[inds]
```
if such a representation exists. Otherwise return `nothing`.

!!! note
    This assumes the selected Cartesian coordinates are unique.
"""
function factorize_indices(x::AbstractArray{<:Any, N}, inds) where {N}
    cart_inds = @views CartesianIndices(x)[inds]
    isempty(cart_inds) && return ntuple(_ -> Int[], Val(N))
    return _factorize_indices(vec(cart_inds))
end
_factorize_indices(cis::AbstractVector{<:CartesianIndex{1}}) = ([ci[1] for ci in cis],)
function _factorize_indices(cis::AbstractVector{<:CartesianIndex{N}}) where {N}
    tail0 = Base.tail(Tuple(first(cis)))
    i1 = firstindex(cis)
    ℓ1 = something(findfirst(ci -> Base.tail(Tuple(ci)) != tail0, cis), lastindex(cis) + 1) - i1

    v1 = [cis[i][1] for i in 1:ℓ1]

    nblocks, r = divrem(length(cis), ℓ1)
    r == 0 || return nothing

    rest = Vector{CartesianIndex{N - 1}}(undef, nblocks)
    for b in 1:nblocks
        offset = i1 - 1 + (b - 1) * ℓ1
        tailb = Base.tail(Tuple(cis[offset + 1]))
        for i in 1:ℓ1
            ci = cis[offset + i]
            ci[1] == v1[i] && Base.tail(Tuple(ci)) == tailb || return nothing
        end
        rest[b] = CartesianIndex(tailb)
    end

    rest_inds = _factorize_indices(rest)
    isnothing(rest_inds) && return nothing
    return (v1, rest_inds...)
end

# work around Distributions.jl not implementing `length` for `ReshapedDistribution`
_reshape(dist::Distributions.Distribution, sz::Tuple) = reshape(dist, sz)
_reshape(dist::Distributions.ReshapedDistribution, sz::Tuple) = reshape(dist.dist, sz)
