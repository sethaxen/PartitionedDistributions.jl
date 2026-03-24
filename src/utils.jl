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
    A_ii = i isa Int ? A[i, i] : view(A, i, i)
    S = PDMats.PDMat(LinearAlgebra.Symmetric(A_ii - A_ic_i' * B))
    return S, B, A_ic_ic
end
# For PDiagMat, off-diagonal blocks are zero: B = 0, S = diagonal submatrix.
function _schur_complement_and_factor(A::PDMats.PDiagMat, i)
    ic = Not(i)
    n_i = length(i)
    n_ic = length(A.diag) - n_i
    Σ_ic = PDMats.PDiagMat(view(A.diag, ic))
    i isa Int && return A.diag[i], FillArrays.Zeros(n_ic), Σ_ic
    return PDMats.PDiagMat(view(A.diag, i)), FillArrays.Zeros(n_ic, n_i), Σ_ic
end
# For ScalMat, off-diagonal blocks are zero: B = 0, S = scalar * I submatrix.
function _schur_complement_and_factor(A::PDMats.ScalMat, i)
    n_i = length(i)
    A_val = first(A)
    A_dim = size(A, 1)
    n_ic = A_dim - n_i
    Σ_ic = PDMats.ScalMat(n_ic, A_val)
    i isa Int && return A_val, FillArrays.Zeros(n_ic), Σ_ic
    return PDMats.ScalMat(n_i, A_val), FillArrays.Zeros(n_ic, n_i), Σ_ic
end

# symmetric submatrix view of an AbstractPDMat, assumes i is not an Int
_pdview(A::PDMats.AbstractPDMat, i) = PDMats.AbstractPDMat(view(A, i, i))
_pdview(A::PDMats.PDiagMat, i) = PDMats.PDiagMat(view(A.diag, i))
_pdview(A::PDMats.ScalMat, i) = PDMats.ScalMat(length(i), first(A))

_mvnormal(dist::Distributions.MvNormal) = dist
_mvnormal(dist::Distributions.AbstractMvNormal) = Distributions.MvNormal(Distributions.mean(dist), Distributions.cov(dist))
