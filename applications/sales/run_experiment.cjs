"use strict";
// Reproduce the original protocol, export every replicate, and verify the prior report.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const m = require('./numerical_core.cjs');
const root = __dirname;
const c = m.CONFIG, d = c.n_items;
const protocol = JSON.parse(fs.readFileSync(path.join(root, 'protocol.json'), 'utf8'));
for (const key of Object.keys(c)) {
  if (JSON.stringify(c[key]) !== JSON.stringify(protocol[key])) throw Error('Protocol mismatch: ' + key);
}
for (const dir of ['data', 'results']) fs.mkdirSync(path.join(root, dir), { recursive: true });
const writeJson = (file, data) => fs.writeFileSync(path.join(root, file), JSON.stringify(data, null, 2) + '\n');
const csvValue = v => /[",\n\r]/.test(String(v)) ? '"' + String(v).replaceAll('"', '""') + '"' : String(v);
function writeCsv(file, rows) {
  const keys = Object.keys(rows[0]);
  fs.writeFileSync(path.join(root, file), keys.map(csvValue).join(',') + '\n' + rows.map(r => keys.map(k => csvValue(r[k])).join(',')).join('\n') + '\n');
}
function exportBaskets(file, codes) {
  fs.writeFileSync(path.join(root, file), c.items.join(',') + '\n' + codes.map(x => Array.from({length:d}, (_,j) => (x >> j) & 1).join(',')).join('\n') + '\n');
}
const truth = new Float64Array(d);
for (const i of c.group) truth[i] = 1 / Math.sqrt(c.group.length);
const rows = [], examples = {}, permutations = {}, datasets = [];
const diagnostics = {max_eigen_residual: 0, max_lanczos_steps: 0};
const started = Date.now();
for (const [condition, theta] of [['planted', c.theta], ['null', 0]]) {
  const D = m.distribution(d, c.group, theta);
  const sizes = condition === 'planted' ? c.sample_sizes : [8192];
  for (let replicate = 0; replicate < c.replicates; replicate++) {
    const seed = c.seed + replicate * 100 + (condition === 'null' ? 100000 : 0);
    const train = m.sample(D, 8192, seed);
    const validation = m.sample(D, c.validation_samples, seed + 1);
    const valraw = m.moments(validation, d);
    datasets.push({condition, replicate, train_seed: seed, validation_seed: seed + 1, train: Array.from(train), validation: Array.from(validation)});
    for (const n of sizes) {
      const fit = m.fit(m.moments(train, d, n), d, seed + 2);
      for (const [method, r] of Object.entries(fit.methods)) {
        rows.push({condition, replicate, seed, n_train: n, method, ...m.metric(r.u, truth, c.group, valraw, d), eigen_residual: r.info.residual});
        diagnostics.max_eigen_residual = Math.max(diagnostics.max_eigen_residual, r.info.residual);
        diagnostics.max_lanczos_steps = Math.max(diagnostics.max_lanczos_steps, r.info.steps || 0);
      }
      if (replicate === 0 && n === 8192) {
        examples[condition] = {seed, directions: Object.fromEntries(Object.entries(fit.methods).map(([k,v]) => [k, Array.from(v.u)])), covariance: Array.from(fit.covariance), coefficients: Array.from(fit.coefficients), spectral_info: Object.fromEntries(Object.entries(fit.methods).map(([k,v]) => [k,v.info]))};
        permutations[condition] = m.permutation(validation, fit.methods['TensorPCA l=4'].u, d, c.permutations, seed + 3);
        exportBaskets('data/' + condition + '_train.csv', Array.from(train));
        exportBaskets('data/' + condition + '_validation.csv', Array.from(validation));
      }
    }
    if ((replicate + 1) % 4 === 0) console.log(`${condition}: ${replicate + 1}/24 seeds; ${((Date.now()-started)/1000).toFixed(1)} seconds`);
  }
}
const result = {config: c, rows, summary: m.summary(rows), examples, permutations, diagnostics,
  elapsed_seconds: (Date.now()-started)/1000,
  provenance: {original_run_date: '2026-11-09', export_run_date: '2026-11-11', reconstructed_from: 'Executed numerical source and seed schedule retained in the conversation; original transient runtime objects were unavailable.', runtime: process.version, platform: process.platform, prng: 'Mulberry32', method: 'Distinct-index Kikuchi, six eigenpairs by absolute eigenvalue, one-particle candidates ranked using training fourth-order contrast'},
  shared_core_sha256: crypto.createHash('sha256').update(fs.readFileSync(path.join(root, '../../src/kikuchi.cjs'))).digest('hex'),
  source_sha256: crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'numerical_core.cjs'))).digest('hex')};
writeJson('results/results.json', result);
writeCsv('results/trials.csv', rows);
writeCsv('results/summary.csv', result.summary);
fs.writeFileSync(path.join(root, 'data/all_replicates_packed.json'), JSON.stringify({encoding: 'For each basket code, item j is (code >> j) & 1. Least significant bit is item 0. Training sample sizes use prefixes of each 8192-row training set.', items: c.items, n_items: d, group_indices: c.group, planted_direction: Array.from(truth), replicates: datasets}) + '\n');
writeCsv('data/items.csv', c.items.map((item,j) => ({index: j, item, planted_weight: truth[j], in_planted_group: +c.group.includes(j)})));
const originalMeans = {
  256: [0.03945478147523239, 0.8184182127613594, 0.6996484801871651],
  1024: [0.07546468212545408, 0.9658692769348362, 0.894696051039387],
  4096: [0.03473362310407156, 0.9936603847691409, 0.9718346539854563],
  8192: [0.0410713538907166, 0.9969056267025458, 0.9826263309973856]
};
const methodOrder = ['PCA', 'TensorPCA l=2', 'TensorPCA l=4'];
let maxError = 0;
for (const s of result.summary.filter(x => x.condition === 'planted')) maxError = Math.max(maxError, Math.abs(s.squared_cosine_mean - originalMeans[s.n_train][methodOrder.indexOf(s.method)]));
const replay = {max_abs_error_against_12_original_reported_means: maxError, tolerance: 1e-8, planted_permutation_p: permutations.planted.p_value, null_permutation_p: permutations.null.p_value};
replay.passed = maxError < 1e-8 && replay.planted_permutation_p === 0.005 && replay.null_permutation_p === 0.96;
writeJson('results/reproduction_check.json', replay);
if (!replay.passed) throw Error('Original reported results were not reproduced: ' + JSON.stringify(replay));
console.log(JSON.stringify({reproduction: replay, diagnostics, elapsed_seconds: result.elapsed_seconds}, null, 2));
