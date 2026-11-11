# Adding an application

Create `applications/<name>/` with a README, a fixed protocol, a data-generation
or data-loading entry point, and an evaluation runner. Convert the application's
observations into the distinct fourth-order coefficients expected by the shared
core, then import `src/kikuchi.cjs` instead of copying the solver.

Keep scientific choices in the application's protocol: preprocessing, sample
unit, sample count, seed schedule, splits, null model, baselines, and metrics.
Training must not receive ground truth or validation data. State whether the
sample distribution satisfies assumptions used in any theoretical argument.

Keep compact example data and reviewed results under the application. Ignore
large reproducible intermediates and explain how to regenerate them. Include
the source hash, protocol, and numerical residuals in recorded results.

When a shared algorithm change affects a previous application, rerun that
application's scientific checks and record the effect. Preserve original
reference results rather than silently retuning a benchmark to pass.

The older `tensorPCA_old` and `brain-tensor-pca` repositories are future migration
sources. No code or data from them has been imported during this setup.
