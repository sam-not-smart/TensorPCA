"""Optional plots, Markdown report, and dataset-export validation. No model fitting."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'results/results.json').read_text())
c = r['config']
rows = r['summary']
methods = ['PCA', 'TensorPCA l=2', 'TensorPCA l=4']
colors = dict(zip(methods, ['#777777', '#E69F00', '#0072B2']))

# Check every easy-to-open CSV against the full packed data export.
packed = json.loads((ROOT / 'data/all_replicates_packed.json').read_text())
assert len(packed['replicates']) == 48
for rep in packed['replicates']:
    for split in ['train', 'validation']:
        assert len(rep[split]) == 8192
        assert all(isinstance(code, int) and 0 <= code < 4096 for code in rep[split])
    if rep['replicate'] != 0:
        continue
    for split in ['train', 'validation']:
        path = ROOT / 'data' / f"{rep['condition']}_{split}.csv"
        with path.open(newline='') as stream:
            reader = csv.reader(stream)
            assert next(reader) == c['items']
            observed = np.array([[int(value) for value in row] for row in reader])
        expected = (np.asarray(rep[split])[:, None] >> np.arange(12)) & 1
        np.testing.assert_array_equal(observed, expected)
(ROOT / 'results/export_check.json').write_text(json.dumps({'passed': True, 'replicates': 48, 'baskets': 786432, 'csv_files_checked': 4}, indent=2))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
for method in methods:
    series = [s for s in rows if s['condition'] == 'planted' and s['method'] == method]
    x = [s['n_train'] for s in series]
    y = np.array([s['squared_cosine_mean'] for s in series])
    sd = np.array([s['squared_cosine_sd'] for s in series])
    axes[0].plot(x, y, 'o-', color=colors[method], label=method)
    axes[0].fill_between(x, np.maximum(0, y-sd), np.minimum(1, y+sd), color=colors[method], alpha=.15)
axes[0].axhline(1/12, color='black', linestyle=':', linewidth=1, label='Uniform random direction')
axes[0].set_xscale('log', base=2)
axes[0].set_xticks(c['sample_sizes'], [f'{n:,}' for n in c['sample_sizes']])
axes[0].set_ylim(-.02, 1.04)
axes[0].set_xlabel('Training baskets')
axes[0].set_ylabel('Squared cosine with planted direction')
axes[0].set_title('Recovery across 24 independent seeds')
axes[0].text(.04, .92, 'Shading: +/- 1 SD across seeds', transform=axes[0].transAxes, fontsize=8)
axes[0].legend(fontsize=8, loc='center right')
truth = np.zeros(12)
truth[c['group']] = 1/np.sqrt(6)
x = np.arange(12)
axes[1].bar(x-.25, truth, width=.25, color='#009E73', label='Planted')
for offset, method in [(0, 'TensorPCA l=4'), (.25, 'PCA')]:
    u = np.array(r['examples']['planted']['directions'][method])
    if u @ truth < 0:
        u = -u
    axes[1].bar(x+offset, u, width=.25, color=colors[method], label=method)
axes[1].set_xticks(x, c['items'], rotation=60, ha='right')
axes[1].set_ylabel('Unit-vector weight')
axes[1].set_title('Predetermined first seed: product weights')
axes[1].legend(fontsize=8)
fig.suptitle('Synthetic sales-group recovery | 12 products, one planted group', fontsize=14)
fig.savefig(ROOT / 'results/recovery.png', dpi=180)
fig.savefig(ROOT / 'results/recovery.pdf')
plt.close(fig)

report = ['# Synthetic sales-group recovery: reproduced results', '',
          'Original experiment: 9 November 2026. Reproduced and exported: 11 November 2026.', '',
          'The original numerical source and seeds were reconstructed from the conversation. '
          'All 12 original planted recovery means and both permutation p-values were checked. '
          'See `reproduction_check.json` for exact discrepancies.', '',
          '## Results at 8,192 training baskets', '',
          '| Method | Mean squared cosine +/- SD | Exact top-six group |',
          '|---|---:|---:|']
for method in methods:
    s = next(s for s in rows if s['condition']=='planted' and s['n_train']==8192 and s['method']==method)
    report.append(f"| {method} | {s['squared_cosine_mean']:.6f} +/- {s['squared_cosine_sd']:.6f} | {round(s['exact_support_mean']*24)}/24 |")
report += ['', 'Squared cosine 1 means perfect recovery. Group size is used only for scoring the top-six ranking.', '',
           '## Sample-size sweep', '', '| Training baskets | PCA | TensorPCA level 2 | TensorPCA level 4 |', '|---:|---:|---:|---:|']
for n in c['sample_sizes']:
    vals = [next(s for s in rows if s['condition']=='planted' and s['n_train']==n and s['method']==method)['squared_cosine_mean'] for method in methods]
    report.append(f'| {n:,} | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
report += ['', '![Recovery and first-seed weights](recovery.png)', '',
           '## Validation and controls', '',
           f"The predetermined first-seed validation permutation test gives p={r['permutations']['planted']['p_value']:.3f} for the planted condition and p={r['permutations']['null']['p_value']:.2f} for the null. Each uses 199 permutations, with a minimum attainable p-value of 0.005.", '',
           '| Condition | Mean level-4 validation contrast +/- SD |', '|---|---:|']
for condition in ['planted','null']:
    s = next(s for s in rows if s['condition']==condition and s['n_train']==8192 and s['method']=='TensorPCA l=4')
    report.append(f"| {condition} | {s['validation_contrast_mean']:.6f} +/- {s['validation_contrast_sd']:.6f} |")
report += ['', f"Maximum eigenpair residual across all runs: {r['diagnostics']['max_eigen_residual']:.3e}.", '',
           'Exact enumeration confirms the matching first-, second-, and third-order population statistics. '
           'Numerical checks compare cumulants with direct centered calculations and Lanczos eigenpairs with a dense solve. '
           'Dataset export checks confirm each of the four CSVs matches its packed replicate.', '',
           '## Interpretation', '',
           'Higher-order statistics recover the deliberately planted product group in this controlled test. '
           'Level 2 performs better than level 4, so these results do not establish an advantage for a larger lift. '
           'The implementation is classical and uses the distinct-index Kikuchi operator. '
           'Retail realism, recommendation improvement, overlapping groups, and quantum advantage are untested. '
           'The README specifies the probability distribution, algorithm, seed schedule, and data format.']
(ROOT / 'results/REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
print('Dataset export validated; report and plots generated.')
