"""Paper-layout summaries of saved BNN measurements; no Theano/training imports."""
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np

METHODS = ('ASVGD', 'SVGD')
METRICS = ('RMSE', 'LL', 'seconds')


def constructor_settings(model, **overrides):
    """Capture actual defaults plus explicit call arguments, without instantiation."""
    values = {name: p.default for name, p in inspect.signature(model).parameters.items()
              if p.default is not inspect.Parameter.empty}
    values.update(overrides)
    return values


def _tex(text):
    replacements = {'\\': r'\textbackslash{}', '_': r'\_', '&': r'\&', '%': r'\%',
                    '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(text))


def write_summary(results, budgets, output_dir, *, expected_datasets=None, settings=None):
    """Write all budgets, with dataset rows and six ASVGD/SVGD metric columns.

    results[dataset][method] has shape (runs, budgets, 3): RMSE, LL, seconds.
    Nonfinite/incomplete arrays are rejected, never omitted with nanmean.
    Exact ties bold both methods; bold is descriptive, not a significance test.
    """
    budgets = list(budgets)
    if (not budgets or any(type(b) is not int or b <= 0 for b in budgets)
            or len(set(budgets)) != len(budgets)):
        raise ValueError('budgets must be distinct positive integers in stored array order')
    if not results:
        raise ValueError('no completed datasets')
    datasets = list(results)
    expected = list(expected_datasets) if expected_datasets is not None else datasets
    if len(set(expected)) != len(expected) or not set(datasets) <= set(expected):
        raise ValueError('completed datasets must be a subset of the unique expected datasets')
    stats, counts = {}, {}
    for dataset in datasets:
        if set(results[dataset]) != set(METHODS):
            raise ValueError(f'{dataset}: both ASVGD and SVGD arrays required')
        shape = None
        for method in METHODS:
            array = np.asarray(results[dataset][method], dtype=float)
            if array.ndim != 3 or array.shape[0] == 0 or array.shape[1:] != (len(budgets), 3):
                raise ValueError(f'{dataset}/{method}: expected (runs, {len(budgets)}, 3)')
            if shape is not None and array.shape != shape:
                raise ValueError(f'{dataset}: methods have unequal array shapes')
            shape = array.shape
            if (not np.all(np.isfinite(array)) or np.any(array[:, :, 0] < 0)
                    or np.any(array[:, :, 2] <= 0)):
                raise ValueError(f'{dataset}/{method}: nonfinite or invalid measurements; no runs dropped')
            stats[dataset, method] = (array.mean(axis=0), array.std(axis=0, ddof=0))
            counts[dataset] = array.shape[0]
    # Validate everything before touching any output.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = set(datasets) == set(expected)
    status = 'complete' if complete else 'partial'
    note = ('Mean +/- population SD (ddof=0); bold marks the better unrounded mean '
            '(RMSE/time lower, LL higher; exact ties both), not statistical significance.')
    sections = []
    rows = []
    files = []
    for k, budget in enumerate(budgets):
        heading = f'BNN summary: iteration budget {budget} ({status}; {len(datasets)}/{len(expected)} datasets)'
        markdown = [f'## {heading}', '',
                    '| Dataset | RMSE ASVGD | RMSE SVGD | LL ASVGD | LL SVGD | seconds ASVGD | seconds SVGD |',
                    '|---|---:|---:|---:|---:|---:|---:|']
        latex = [r'\begin{table}[htp]', r'\centering', r'\setlength{\tabcolsep}{8pt}',
                 r'\renewcommand{\arraystretch}{1.2}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
                 r' & \multicolumn{2}{c}{RMSE ($\downarrow$)} & \multicolumn{2}{c}{LL ($\uparrow$)} & \multicolumn{2}{c}{Seconds ($\downarrow$)} \\',
                 r'Dataset & ASVGD & SVGD & ASVGD & SVGD & ASVGD & SVGD \\', r'\midrule']
        for dataset in datasets:
            md_cells, tex_cells = [], []
            for metric, name in enumerate(METRICS):
                means = [stats[dataset, method][0][k, metric] for method in METHODS]
                best = max(means) if name == 'LL' else min(means)
                for method, mean in zip(METHODS, means):
                    sd = stats[dataset, method][1][k, metric]
                    winner = bool(mean == best)
                    cell = f'{mean:.3f} +/- {sd:.3f}'
                    md_cells.append(f'**{cell}**' if winner else cell)
                    tex_cell = f'{mean:.3f}_{{\\pm {sd:.3f}}}'
                    tex_cells.append('$' + (r'\mathbf{' + tex_cell + '}' if winner else tex_cell) + '$')
                    rows.append([dataset, budget, method, name, counts[dataset], float(mean), float(sd), winner])
            markdown.append('| ' + dataset.replace('|', r'\|') + ' | ' + ' | '.join(md_cells) + ' |')
            latex.append(_tex(dataset) + ' & ' + ' & '.join(tex_cells) + r' \\')
        run_note = 'Runs per dataset: ' + ', '.join(f'{d}={counts[d]}' for d in datasets) + '.'
        markdown.extend(['', note, run_note, ''])
        latex.extend([r'\bottomrule', r'\end{tabular}',
                      r'\caption{' + _tex(heading + '. ' + note + ' ' + run_note) + '}', r'\end{table}', ''])
        md_path, tex_path = output_dir/f'summary_{budget}.md', output_dir/f'summary_{budget}.tex'
        md_path.write_text('\n'.join(markdown), encoding='utf-8')
        tex_path.write_text('\n'.join(latex), encoding='utf-8')
        files.extend([str(md_path), str(tex_path)])
        sections.append('\n'.join(markdown))
    csv_path = output_dir/'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['dataset', 'iterations', 'method', 'metric', 'runs', 'mean', 'population_sd', 'better_mean'])
        writer.writerows(rows)
    metadata = {'status': status, 'datasets_completed': datasets, 'datasets_expected': expected,
                'budgets_in_stored_order': budgets, 'runs_per_dataset': counts, 'sd_ddof': 0,
                'metric_order_in_raw_arrays': list(METRICS), 'settings': settings,
                'note': note, 'files': files + [str(csv_path)]}
    (output_dir/'summary_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (output_dir/'summary.md').write_text('\n'.join(sections), encoding='utf-8')
    return '\n'.join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', required=True,
                        help='Iteration budgets in the EXACT stored array order')
    parser.add_argument('--datasets', nargs='+', help='Default: all datasets with either method file')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    datasets = args.datasets
    if datasets is None:
        datasets = sorted({p.stem[len(f'results_{m}_'):] for m in METHODS
                           for p in args.results.glob(f'results_{m}_*.npy')})
    results = {d: {m: np.load(args.results/f'results_{m}_{d}.npy', allow_pickle=False)
                   for m in METHODS} for d in datasets}
    print(write_summary(results, args.budgets, args.output or args.results/'bnn_summary',
                        settings={'source': str(args.results), 'training_settings': 'not supplied'}))


if __name__ == '__main__':
    main()
