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
