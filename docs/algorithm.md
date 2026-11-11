# Distinct-index fourth-order Kikuchi core

The shared core accepts a symmetric fourth-order tensor restricted to distinct
indices. Its coefficient vector is ordered by lexicographic four-subsets of
`{0,...,d-1}`. Repeated-index entries are not supplied or inferred.

For lift level ell, basis states are ell-subsets. The real symmetric operator is

\[
K_{U,V}=\begin{cases}
T_{U\triangle V}, & |U\triangle V|=4,\\
0, & \text{otherwise}.
\end{cases}
\]

Its dimension is `choose(d,ell)`, row degree is
`choose(ell,2)*choose(d-ell,2)`, and squared Frobenius norm is
`6*choose(d-4,ell-2)*sum(T_S^2)`. The trace is zero. Do not normalize eigenvalue
concentration by the sum of eigenvalues.

The code obtains six eigenpairs by largest absolute eigenvalue. Small matrices
use a dense Jacobi solve; larger matrices use Lanczos with full reorthogonalization
and checked Ritz residuals. Exact termination of the Krylov space is accepted,
including before 40 iterations. Such a termination can provide fewer than six
independent Ritz vectors for a degenerate operator; the available vectors are used.

For each lifted eigenvector z, form `A[W,a] = z[W union {a}]` when a is not in W,
and zero otherwise, where W is an (ell-1)-subset. The one-particle matrix is
`A.T*A/ell`. Its eigenvectors are candidate recovered directions.

Each candidate u is ranked by the absolute projection onto the input tensor of
its normalized distinct-index product pattern `prod(u_i for i in S)`.
The highest training contrast wins. Global vector sign is unidentifiable for
an even-order tensor. Recovery accuracy is assessed outside the core.

`recoverDirection(coefficients, {dimension, level, seed})` returns a unit vector
in `direction` and spectral diagnostics in `info`. It rejects non-finite inputs,
incorrect vector lengths, and the zero tensor. Subset masks require dimension
between 4 and 30, but this is a representation limit, not a practical scalability
claim: building the operator and Krylov vectors becomes costly much earlier.

Current tests cover a signed, equal-magnitude six-mode spike, numerical operator
identities, comparison with a dense eigensolve, and the full sampled sales
experiment. They are not a guarantee for arbitrary tensors or all input scales.
The numerical routines retain the validated prototype's absolute tolerances.

Hastings' classical and quantum spectral work provides the background:
https://arxiv.org/abs/1907.12724. This subset operator is distinguished from the
full bosonic occupation-number construction, whose dimension permits repeated
mode indices. No quantum runtime bound is claimed for this implementation or
the synthetic basket distribution.
