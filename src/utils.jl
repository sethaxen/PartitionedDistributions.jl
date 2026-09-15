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
_reshape(dist::Distributions.UnivariateDistribution, ::Tuple{}) = dist
_reshape(dist::Distributions.ReshapedDistribution, sz::Tuple) = _reshape(dist.dist, sz)
function _reshape(  # resolve ambiguity
        dist::Distributions.ReshapedDistribution{0, S, D},
        ::Tuple{},
    ) where {
        S <: Distributions.ValueSupport,
        D <: Distributions.Distribution{<:Distributions.ArrayLikeVariate, S},
    }
    return _reshape(dist.dist, ())
end

# Helpers shared by the pointwise log-pdf functions

function _logpdf_eltype(dist::Distributions.Distribution, x)
    return typeof(log(one(promote_type(eltype(x), Distributions.partype(dist)))))
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

function _similar_logpdf(dist::Distributions.UnivariateDistribution, x::Number)
    return zero(_logpdf_eltype(dist, x))
end
function _similar_logpdf(
        dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate}, x
    )
    return similar(x, _logpdf_eltype(dist, x))
end
function _similar_logpdf(
        dist::Distributions.ProductNamedTupleDistribution, x::NamedTuple{K}
    ) where {K}
    return map(_similar_logpdf, NamedTuple{K}(dist.dists), x)
end

# diag(inv(A)) without forming the full inverse
function _pd_diag_inv(A::PDMats.AbstractPDMat)
    T = typeof(float(oneunit(eltype(A))))
    I = LinearAlgebra.Diagonal(ones(T, axes(A, 1)))
    return PDMats.invquad(A, I)
end

# Logarithms of the modified Bessel functions, log(I_ν(t)) and log(K_ν(t)), for t > 0.
#
# The exponentially scaled functions from SpecialFunctions.jl are used whenever they can be
# evaluated without underflow (besselix, for t ≪ ν) or overflow (besselkx, for t ≪ ν). Otherwise,
# for large orders, the uniform asymptotic expansions of DLMF 10.41.3 and 10.41.4 with 5 terms
# are used; their absolute error in the log is about 2e-7 for ν = 10 and decays as ν^-5.

const _BESSEL_ASYMPTOTIC_MIN_ORDER = 10
# AMOS signals overflow/underflow of the scaled functions at a log-magnitude of about 699-701,
# slightly inside log(floatmax(Float64)) ≈ 709.8, so keep a margin.
const _BESSEL_SCALED_LOG_LIMIT = 690.0

function _logbesseli(ν::Real, t::Real)
    T = float(promote_type(typeof(ν), typeof(t)))
    return T(_logbesseli(Float64(ν), Float64(t)))
end
function _logbesseli(ν::Float64, t::Float64)
    ν < _BESSEL_ASYMPTOTIC_MIN_ORDER && return log(SpecialFunctions.besselix(ν, t)) + t
    logi = _logbesseli_asymptotic(ν, t)
    # besselix(ν, t) = exp(-t) I_ν(t) ≤ 1 cannot overflow
    logi - t > -_BESSEL_SCALED_LOG_LIMIT && return log(SpecialFunctions.besselix(ν, t)) + t
    return logi
end

function _logbesselk(ν::Real, t::Real)
    T = float(promote_type(typeof(ν), typeof(t)))
    return T(_logbesselk(Float64(ν), Float64(t)))
end
function _logbesselk(ν::Float64, t::Float64)
    ν < _BESSEL_ASYMPTOTIC_MIN_ORDER && return log(SpecialFunctions.besselkx(ν, t)) - t
    logk = _logbesselk_asymptotic(ν, t)
    # besselkx(ν, t) = exp(t) K_ν(t) ≥ sqrt(π / 2t) cannot underflow
    logk + t < _BESSEL_SCALED_LOG_LIMIT && return log(SpecialFunctions.besselkx(ν, t)) - t
    return logk
end

_logbesseli_asymptotic(ν, t) = _logbessel_uniform_asymptotic(ν, t, 1) - log(2π * ν) / 2
_logbesselk_asymptotic(ν, t) = _logbessel_uniform_asymptotic(ν, t, -1) + log(π / (2ν)) / 2
# shared part of DLMF 10.41.3 (sign = 1) and 10.41.4 (sign = -1) with t = ν z
function _logbessel_uniform_asymptotic(ν, t, sign)
    z = t / ν
    s = hypot(one(z), z)  # sqrt(1 + z²)
    p = inv(s)
    η = s + log(z / (1 + s))
    u1, u2, u3, u4 = _bessel_debye_polynomials(p)
    series = 1 + sign * u1 / ν + u2 / ν^2 + sign * u3 / ν^3 + u4 / ν^4
    return sign * ν * η - log(s) / 2 + log(series)
end
# U_1 to U_4 from DLMF 10.41.10
function _bessel_debye_polynomials(p)
    p2 = p^2
    u1 = p * (3 - 5p2) / 24
    u2 = p2 * (81 - 462p2 + 385p2^2) / 1152
    u3 = p * p2 * (30375 - 369603p2 + 765765p2^2 - 425425p2^3) / 414720
    u4 = p2^2 * (4465125 - 94121676p2 + 349922430p2^2 - 446185740p2^3 + 185910725p2^4) / 39813120
    return u1, u2, u3, u4
end
