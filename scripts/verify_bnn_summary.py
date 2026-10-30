#!/usr/bin/env python3
"""No-training tests for BNN summaries and source-preserving integration."""
import ast
import csv
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'legacy/bnn'))
from bnn_summary import constructor_settings, write_summary


def main():
    source_path = ROOT/'legacy/bnn/bayesian_nn_ASVGD.py'
    source = source_path.read_text()
    # Hash of the newline-normalized source at published commit 533d70d.
    # Unlike git-show at runtime, this also works in shallow clones/source archives.
    baseline_sha256 = 'c2845f5f8f586392c2282d4ac38caa55072dd13877ed2ff8dec71358e3bfd483'
    removed = re.sub(r'^ *# BEGIN summary-only addition\n.*?^ *# END summary-only addition\n\n?',
                     '', source, flags=re.MULTILINE | re.DOTALL)
    assert hashlib.sha256(removed.encode()).hexdigest() == baseline_sha256, 'non-summary source changed'
    ast.parse(source)
    a = np.array([[[2., -2., 1.], [1., -1., 2.]], [[4., -4., 3.], [3., -3., 4.]]])
    b = a.copy(); b[:, :, 0] += 1; b[:, :, 1] += 1
    raw = {'data_set&one': {'ASVGD': a, 'SVGD': b}}
    a_before, b_before = a.copy(), b.copy()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)/'summaries'
        text = write_summary(raw, [2000, 5000], output, expected_datasets=['data_set&one', 'other'])
        metadata = json.loads((output/'summary_metadata.json').read_text())
        assert metadata['status'] == 'partial' and metadata['sd_ddof'] == 0
        tex = (output/'summary_2000.tex').read_text()
        assert r'data\_set\&one' in tex
        assert tex.index(r'\caption') > tex.index(r'\end{tabular}')
        assert r'\toprule' in tex and r'\hline' not in tex
        assert '**3.000 +/- 1.000**' in text  # RMSE ASVGD, NOT standard error
        with (output/'summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12
        assert float(rows[0]['mean']) == 3 and float(rows[0]['population_sd']) == 1
        # LL winner is SVGD (less negative); time exact ties bold both.
        assert rows[2]['better_mean'] == 'False' and rows[3]['better_mean'] == 'True'
        assert rows[4]['better_mean'] == rows[5]['better_mean'] == 'True'
        write_summary(raw, [2000, 5000], output)
        assert json.loads((output/'summary_metadata.json').read_text())['status'] == 'complete'
        invalid = [({}, [2000, 5000]), (raw, [2000, 2000]), (raw, [0, 5000]),
                   ({'d': {'ASVGD': a}}, [2000, 5000]),
                   ({'d': {'ASVGD': a, 'SVGD': b[:1]}}, [2000, 5000])]
        for value in (np.nan, np.inf, -1.):
            bad = a.copy(); bad[0, 0, 2] = value
            invalid.append(({'d': {'ASVGD': bad, 'SVGD': b}}, [2000, 5000]))
        for result, budgets in invalid:
            try:
                write_summary(result, budgets, output)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid data accepted')
        # Saved-array CLI needs NumPy only; never imports or instantiates Theano.
        for method, array in raw['data_set&one'].items():
            np.save(Path(tmp)/f'results_{method}_toy.npy', array)
        subprocess.run([sys.executable, str(ROOT/'legacy/bnn/bnn_summary.py'),
                        '--results', tmp, '--budgets', '2000', '5000'], check=True,
                       stdout=subprocess.DEVNULL)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    class Dummy:
        def __init__(self, data, M=20, beta=None):
            raise AssertionError('must not construct models')
    assert constructor_settings(Dummy, M=10) == {'M': 10, 'beta': None}
    # Execute the actual inserted end-of-dataset hook with completed synthetic
    # arrays and constructors that FAIL if called: integration without training.
    hooks = re.findall(r'# BEGIN summary-only addition\n(.*?) *# END summary-only addition',
                       source, flags=re.DOTALL)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as console:
        namespace = dict(completed_results={}, dataset='toy', datasets=['toy'],
                         results_asvgd=a, results_svgd=b, maxiters=[2000, 5000],
                         train_ratio=.9, n_runs=2, batch_size=100, n_hidden=50,
                         svgd_bayesnn=Dummy, asvgd_bayesnn=Dummy, Path=Path,
                         write_summary=write_summary, constructor_settings=constructor_settings)
        with patch.object(Path, 'cwd', return_value=Path(tmp)):
            exec(compile(textwrap.dedent(hooks[1]), '<actual summary hook>', 'exec'), namespace)
        metadata = json.loads((Path(tmp)/'bnn_summary/summary_metadata.json').read_text())
        assert metadata['status'] == 'complete'
        assert metadata['settings']['SVGD']['max_iter'] == [2000, 5000]
        assert 'iteration budget 5000' in console.getvalue()
    assert 'theano' not in sys.modules
    print(json.dumps({'status': 'pass', 'training_source_unchanged_outside_summary_hooks': True,
                      'aggregation': 'mean and population SD, ddof=0',
                      'checks': ['partial/completion', 'all budgets', 'winner directions/ties',
                                 'LaTeX escaping/booktabs/caption', 'CSV precision', 'reject invalid arrays',
                                 'no mutation', 'saved-array CLI', 'defaults without model construction',
                                 'actual end-of-dataset hook with mock constructors'],
                      'training_runs': 0}, indent=2))


if __name__ == '__main__':
    main()
