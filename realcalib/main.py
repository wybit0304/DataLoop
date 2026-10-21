"""GP + expected-improvement identification; main calls vehicle_model.py."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
from realcalib.vehicle_model import (
    NOMINAL, PARAM_NAMES, CAL_COLUMNS, initial_state, rollout_cal, predict_cal,
    cal_target, read_csv, save_csv, write_json, scores, plot_series,
)
SEARCH_NAMES = ['rolling_resistance', 'ix', 'iz', 'ms', 'k_phi_front', 'k_phi_rear', 'c_phi', 'friction', 'steering_scale', 'steering_bias', 'steering_tau', 'gain_fy']

BOUNDS = np.array([[0.001, 0.1], [100, 2000], [100, 10000], [100, 1737], [1000, 30000], [1000, 30000], [100, 8000], [0.1, 1.5], [0.7, 1.3], [-0.02, 0.02], [0.005, 0.15], [1.5, 3.2]])

def unpack(theta):
    p = NOMINAL.copy()
    for i, name in enumerate(SEARCH_NAMES[:7]):
        p[PARAM_NAMES.index(name)] = theta[i]
    p[20] = theta[11]
    return (p, float(theta[7]), {'steering_scale': float(theta[8]), 'steering_bias': float(theta[9]), 'steering_tau': float(theta[10])})

def nominal_theta():
    return np.r_[[NOMINAL[PARAM_NAMES.index(n)] for n in SEARCH_NAMES[:7]], 0.8, 1.0, 0.0, 0.03, 2.6]

def make_cal_demo(folder, seed=42):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = nominal_theta()
    truth[[0, 2, 7, 8, 11]] = [0.022, 3200.0, 0.72, 1.06, 2.3]
    p, mu, act = unpack(truth)
    manifest = {'kind': 'synthetic', 'not_manuscript_data': True, 'generator_seed': seed, 'train': ['train_0.csv', 'train_1.csv'], 'validation': ['validation.csv'], 'test': ['test.csv'], 'true_parameters_for_demo_only': dict(zip(SEARCH_NAMES, truth.tolist()))}
    for i, name in enumerate(['train_0', 'train_1', 'validation', 'test']):
        t = np.arange(0, 12.01, 0.04)
        steering = 0.028 * np.sin((0.8 + i * 0.12) * t) * np.sin(np.pi * t / 12) ** 2
        total = 150 + 60 * np.sin(0.5 * t + i)
        torques = np.tile(total[:, None] / 4, (1, 4))
        torques[:, [1, 3]] += 12 * np.sin(t)[:, None]
        torques[:, [0, 2]] -= 12 * np.sin(t)[:, None]
        y, x = rollout_cal(t, steering, torques, initial_state(9 + i * 0.7, p), p, mu, act['steering_scale'], act['steering_bias'], act['steering_tau'])
        y += rng.normal(size=y.shape) * np.array([0.005, 0.01, 0.0005])
        save_csv(folder / (name + '.csv'), np.column_stack([t, steering, torques, y, x[:, 1]]), CAL_COLUMNS + ['vy_mps'])
    write_json(folder / 'manifest.json', manifest)

def discrepancy(data, theta, scales):
    p, mu, act = unpack(theta)
    pred, _ = predict_cal(data, p, mu, act)
    err = (pred - cal_target(data)) / scales
    if not np.isfinite(err).all() or np.abs(err).max() > 10000.0:
        return 1000000000.0
    e = np.sqrt(np.mean(err ** 2, axis=0))
    return float(e.mean() + 0.2 * e.max() + 0.1 * np.quantile(np.abs(err), 0.95))

def bayesian_optimize(train, trials, seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from scipy.special import ndtr
    import warnings
    rng = np.random.default_rng(seed)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    start = (nominal_theta() - BOUNDS[:, 0]) / span
    scales = np.maximum(np.std(np.concatenate([cal_target(d) for d in train]), axis=0), [0.2, 0.2, 0.02])
    X = []
    Y = []
    for k in range(trials):
        if k == 0:
            z = start.copy()
        elif k < 12:
            z = np.clip(start + rng.normal(0, 0.12, len(start)), 0, 1)
        else:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0, 'fixed') * Matern(length_scale=np.ones(len(start)) * 0.3, nu=2.5), alpha=1e-05, normalize_y=True, optimizer=None, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gp.fit(np.array(X), np.array(Y))
            incumbent = np.array(X)[int(np.argmin(Y))]
            cand = np.vstack([rng.uniform(0, 1, (600, len(start))), np.clip(incumbent + rng.normal(0, 0.07, (600, len(start))), 0, 1)])
            mean, std = gp.predict(cand, return_std=True)
            imp = min(Y) - mean - 0.005
            zeta = imp / np.maximum(std, 1e-12)
            ei = imp * ndtr(zeta) + std * np.exp(-0.5 * zeta * zeta) / np.sqrt(2 * np.pi)
            z = cand[int(np.argmax(ei))]
        theta = BOUNDS[:, 0] + z * span
        val = float(np.mean([discrepancy(d, theta, scales) for d in train]))
        X.append(z)
        Y.append(val)
        if (k + 1) % 10 == 0:
            print(f'GP-EI trial {k + 1}/{trials}: best training loss={min(Y):.5f}', flush=True)
    return (BOUNDS[:, 0] + np.array(X)[int(np.argmin(Y))] * span, scales, {'normalized_parameters': np.array(X).tolist(), 'train_objective': Y})

def model_main():
    ap = argparse.ArgumentParser(description='RealCalib: full 8DOF simulator and GP-EI calibration')
    ap.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    ap.add_argument('--demo', action='store_true', help='Generate synthetic data if manifest is absent')
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    started = time.perf_counter()
    if args.trials < 1:
        ap.error('--trials must be positive')
    if args.demo and (not (args.data_dir / 'manifest.json').exists()):
        make_cal_demo(args.data_dir, args.seed)
    manifest = json.loads((args.data_dir / 'manifest.json').read_text(encoding='utf-8'))
    splits = {k: manifest[k] for k in ['train', 'validation', 'test']}
    paths = [str((args.data_dir / n).resolve()) for seq in splits.values() for n in seq]
    if len(set(paths)) != len(paths) or any((not splits[k] for k in splits)):
        raise ValueError('Separate nonempty train/validation/test files required')
    data = {k: [read_csv(args.data_dir / n, CAL_COLUMNS) for n in seq] for k, seq in splits.items()}
    theta, scales, history = bayesian_optimize(data['train'], args.trials, args.seed)
    p, mu, act = unpack(theta)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'calibrated_parameters.json', {'physical_parameters': dict(zip(PARAM_NAMES, p.tolist())), 'friction': mu, 'actuator': act, 'data_kind': manifest.get('kind', 'author_supplied'), 'scope': 'core-method demonstration; example parameter bounds'})
    result = {'data_kind': manifest.get('kind', 'author_supplied'), 'optimizer': 'Gaussian process / expected improvement', 'seed': args.seed, 'trials': args.trials, 'normalization_fit': 'training only', 'scales': scales.tolist(), 'searched_bounds': dict(zip(SEARCH_NAMES, BOUNDS.tolist())), 'splits': splits, 'evaluation': {}}
    for split, arrs in data.items():
        result['evaluation'][split] = []
        for j, d in enumerate(arrs):
            target = cal_target(d)
            pred, _ = predict_cal(d, p, mu, act)
            baseline, _ = predict_cal(d, *unpack(nominal_theta()))
            result['evaluation'][split].append({'file': splits[split][j], 'nominal': scores(target, baseline, ['vx_mps', 'ay_mps2', 'yaw_rate_radps']), 'calibrated': scores(target, pred, ['vx_mps', 'ay_mps2', 'yaw_rate_radps'])})
            save_csv(args.out / f'{split}_{j}_prediction.csv', np.column_stack([d['time_s'], target, pred]), ['time_s'] + ['ref_' + n for n in CAL_COLUMNS[-3:]] + ['pred_' + n for n in CAL_COLUMNS[-3:]])
            if split == 'test':
                plot_series(args.out / f'test_{j}.png', d['time_s'], target, pred, CAL_COLUMNS[-3:])
    result['runtime_seconds'] = time.perf_counter() - started
    result['scope'] = 'core-method demonstration; nominal/demo parameters'
    write_json(args.out / 'metrics.json', result)
    write_json(args.out / 'optimization_history.json', history)
    print(json.dumps(result['evaluation']['test'], indent=2))
    print(f'Outputs: {args.out.resolve()}')

if __name__ == '__main__':
    model_main()
