# TensorPCA

A shared TensorPCA/Kikuchi implementation with reproducible application experiments.

The first application recovers a planted higher-order product group from synthetic
sales baskets. The algorithm lives in `src/`; application-specific data generation,
preprocessing, evaluation, and reports live in `applications/`.

## Run the core

Requires Node.js 20 or newer; validated with Node.js 24. No npm dependencies or
installation step is required.

```sh
node examples/planted_tensor.cjs
node --test tests/kikuchi.test.cjs
```

The example constructs a distinct fourth-order tensor and recovers its planted
direction without using the sales application.

```javascript
const { recoverDirection, quartetIndices } = require('./src/kikuchi.cjs');

const dimension = 12;
const coefficients = quartetIndices(dimension).map(
  quartet => quartet.every(index => index < 6) ? 0.12 : 0
);
const { direction, info } = recoverDirection(coefficients, {
  dimension,
  level: 2,
  seed: 7,
});
console.log(Array.from(direction));
console.log(info.residual);
```

Supply coefficients for `i < j < k < l` in the lexicographic order returned by
`quartetIndices(dimension)`. The output direction has unit norm and an arbitrary
global sign. The API deliberately receives no product labels or planted target.

## Run the sales application

```sh
node applications/sales/test_numerics.cjs
node applications/sales/run_experiment.cjs
```

The complete run covers 24 planted and 24 control seeds, four training-set sizes,
PCA and two Kikuchi levels, independent validation, and permutation tests. It
also verifies the recovery averages from the original experiment. Smaller
training sets use prefixes of the same replicate's training data.

Optional figures and the report use Python with NumPy and Matplotlib:

```sh
python -m pip install -r applications/sales/requirements.txt
python applications/sales/make_report.py
```

See [the sales README](applications/sales/README.md) for the exact probability
model and data schema, and [the results report](applications/sales/results/REPORT.md)
for the recorded findings.

## Layout

| Path | Purpose |
|---|---|
| `src/kikuchi.cjs` | Shared fourth-order operator, eigensolvers, and one-particle recovery |
| `examples/planted_tensor.cjs` | Small application-independent run |
| `tests/` | Core API and recovery tests |
| `applications/sales/` | Sales experiment, data, protocol, tests, and results |
| `docs/algorithm.md` | Mathematical conventions and implementation limits |
| `docs/adding-applications.md` | Convention for introducing another application |
| `.github/workflows/tests.yml` | Core tests and full sales reproduction on GitHub |

The present implementation is dependency-free JavaScript because it preserves
the validated sales experiment. Python is currently used only for plotting and
report generation. Further language bindings can be added without changing the
experiment protocol.

## Current scope

This is a classical, distinct-index fourth-order Kikuchi implementation. It is
not the full repeated-index bosonic Hamiltonian or a quantum circuit simulator.
The operator grows combinatorially, so the current numerical code is intended
for small research experiments.

On the synthetic sales test, level 2 recovers the planted direction more
accurately than level 4. These results do not establish a larger-lift or quantum
advantage, or improved recommendations on real retail data.

Migration from `tensorPCA_old` and `brain-tensor-pca` is deferred. Neither older
repository is required to run this one. No license has been selected yet.
