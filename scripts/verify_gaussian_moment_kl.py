#!/usr/bin/env python3
"""Formula checks and old/new trajectory regression; no performance conclusions."""
from dataclasses import replace
from decimal import Decimal, localcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsi_reproducer.algorithms import METHODS, run_experiment
from gsi_reproducer.configs import PAPER_CONFIGS
from gsi_reproducer.metrics import gaussian_moment_kl, metric_name
from gsi_reproducer.targets import GAUSSIAN, NON_LIPSCHITZ, TARGETS
from run import save_result

BASELINE = "dff7310bc6ecef5ca024598f815ac0f4249294de"


def main():
    rng = np.random.default_rng(20261030)
    errors, affine_errors = [], []
    for d in (1, 2, 3, 5):
        for _ in range(25):
            r = rng.normal(size=(d, d))
            q = r @ r.T + np.eye(d)
            target = replace(GAUSSIAN, mean=rng.normal(size=d), covariance=q)
            x = rng.normal(size=(40, d)) @ (r + 2*np.eye(d)) + rng.normal(size=d)
            value = gaussian_moment_kl(x, target)
            mu, cov = x.mean(axis=0), np.atleast_2d(np.cov(x, rowvar=False, bias=True))
            delta = mu - target.mean
            direct = .5*(np.trace(np.linalg.solve(q, cov)) + delta @ np.linalg.solve(q, delta)
                         - d + np.linalg.slogdet(q)[1] - np.linalg.slogdet(cov)[1])
            errors.append(abs(value-direct) / max(1, abs(direct)))
            t = rng.normal(size=(d, d)) + 3*np.eye(d)
            shift = rng.normal(size=d)
            qt = t @ q @ t.T
            qt = (qt + qt.T)/2
            transformed = replace(target, mean=t @ target.mean+shift, covariance=qt)
            affine_errors.append(abs(gaussian_moment_kl(x @ t.T+shift, transformed)-value)
                                 / max(1, value))
            assert value >= 0
    assert max(errors) < 1e-10, errors
    assert max(affine_errors) < 1e-9, affine_errors

    target = replace(GAUSSIAN, mean=np.zeros(2), covariance=np.eye(2))
    cloud = np.sqrt(2)*np.concatenate((np.eye(2), -np.eye(2)))
    assert gaussian_moment_kl(cloud, target) < 1e-28
    np.testing.assert_allclose(gaussian_moment_kl(cloud+np.array([1., 2.]), target), 2.5)
    # Singular fit is infinity, not jittered or reported as convergence.
    for x in (np.ones((8, 2)), np.zeros((1, 2)), np.array([[0., 0.], [1., 2.], [2., 4.]])):
        assert np.isposinf(gaussian_moment_kl(x, target))
    for x, t in ((cloud, NON_LIPSCHITZ), (np.array([[np.nan, 0.]]), target),
                 (np.empty((0, 2)), target), (cloud[:, 0], target),
                 (cloud, replace(target, covariance=np.diag([-1., -1.]))),
                 (cloud, replace(target, covariance=np.array([[1., 1.], [0., 1.]])))):
        try:
            gaussian_moment_kl(x, t)
        except (ValueError, np.linalg.LinAlgError):
            pass
        else:
            raise AssertionError("invalid input accepted")
    scale = 1 + 1e-8
    with localcontext() as context:
        context.prec = 60
        a = Decimal.from_float(scale)**2
        expected = float(a - 1 - a.ln())
    near = gaussian_moment_kl(scale*cloud, target)
    np.testing.assert_allclose(near, expected, rtol=1e-7, atol=0)
    assert metric_name(TARGETS['anisotropic_gaussian'], 'rbf') == 'historical_metric'
    assert metric_name(NON_LIPSCHITZ, 'bilinear') == 'historical_metric'

    # Execute the unedited, pinned algorithm module for a real baseline comparison.
    source = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{BASELINE}:gsi_reproducer/algorithms.py'], text=True)
    baseline = types.ModuleType('gsi_reproducer._baseline')
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, '<pinned baseline>', 'exec'), baseline.__dict__)
    configurations = [replace(c, particles=12, steps=4) for c in PAPER_CONFIGS.values()]
    configurations.append(replace(configurations[0], name='gaussian_bilinear_Qinv',
                                  bilinear_matrix=((.6, .4), (.4, .6))))
    comparisons = []
    for config in configurations:
        t = TARGETS[config.target_name]
        old = baseline.run_experiment(config, t, progress=False)
        if config.kernel == 'bilinear':
            with patch('gsi_reproducer.metrics.gaussian_kde', side_effect=AssertionError('KDE called')):
                new = run_experiment(config, t, progress=False)
                with tempfile.TemporaryDirectory() as directory:
                    summary = save_result(new, t, Path(directory), save_trajectories=True)
                    assert 'independent_final_kde_kl' not in summary
                    assert summary['covariance_divisor'] == 'N'
                    with np.load(Path(directory)/'results.npz', allow_pickle=False) as arrays:
                        assert 'gaussian_moment_kl_ASVGD' in arrays
                        assert 'historical_ASVGD' not in arrays
        else:
            new = run_experiment(config, t, progress=False)
        for method in METHODS:
            np.testing.assert_array_equal(old.trajectories[method], new.trajectories[method])
            np.testing.assert_array_equal(old.final_states[method], new.final_states[method])
            if config.kernel != 'bilinear':
                np.testing.assert_array_equal(old.historical_metrics[method], new.metrics[method])
            else:
                expected = [gaussian_moment_kl(x, t) for x in new.trajectories[method]]
                np.testing.assert_array_equal(new.metrics[method], expected)
        assert old.failure_iterations == new.failure_iterations
        assert old.restart_counts == new.restart_counts
        comparisons.append(config.name)
    print(json.dumps({'status': 'pass', 'random_formula_cases': 100,
                      'max_relative_formula_error': max(errors),
                      'max_relative_affine_error': max(affine_errors),
                      'near_equilibrium_energy': near,
                      'bitwise_trajectory_regressions': comparisons,
                      'Gaussian_KDE_calls': 0, 'baseline': BASELINE}, indent=2))


if __name__ == '__main__':
    main()
