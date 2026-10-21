"""DeepPhy: eight-DOF UKF prior + causal three-output residual TCN."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from numba import njit
from realcalib.vehicle_model import (
    NOMINAL, initial_state, propagate, dynamics, read_csv, save_csv,
)
SENSOR_COLUMNS = ['time_s', 'steering_rad', 'torque_fl_nm', 'torque_fr_nm', 'torque_rl_nm', 'torque_rr_nm', 'ax_mps2', 'ay_mps2', 'yaw_rate_sensor_radps', 'omega_fl_radps', 'omega_fr_radps', 'omega_rl_radps', 'omega_rr_radps']
REF_COLUMNS = ['vx_ref_mps', 'vy_ref_mps', 'yaw_rate_ref_radps']
STATE_CONFIG = {'window': 50, 'hidden_channels': 64, 'levels': 4, 'kernel_size': 3, 'dropout': 0.1, 'learning_rate': 0.001, 'batch_size': 128, 'epochs': 25, 'lambda_beta': 0.1, 'lambda_residual': 0.001, 'UKF_alpha': 0.3, 'UKF_beta': 2.0, 'UKF_kappa': 0.0, 'min_vx_for_beta': 0.5}

def positive_cov(P):
    P = (P + P.T) / 2
    value, vec = np.linalg.eigh(P)
    return vec * np.maximum(value, 1e-09) @ vec.T

@njit(cache=False)
def sigma_predict(points, steer, torque, mu, p, dt):
    states = np.empty_like(points)
    obs = np.empty((len(points), 7))
    for j in range(len(points)):
        states[j] = propagate(points[j], steer, torque, mu, p, dt)
        _, ax, ay, _ = dynamics(states[j], steer, torque, mu, p)
        obs[j, 0] = ax
        obs[j, 1] = ay
        obs[j, 2] = states[j, 2]
        obs[j, 3:] = states[j, 5:]
    return (states, obs)

def ukf_branch(data, p, mu):
    """Nine-state UKF with IMU, gyro and four wheel-speed measurements.

    No reference velocity or reference sideslip is read in this function.
    The initial forward-speed prior comes only from wheel measurements.
    """
    n = 9
    alpha = 0.3
    lam = alpha ** 2 * n - n
    weights = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    weights[0] = lam / (n + lam)
    wc = weights.copy()
    wc[0] += 1 - alpha ** 2 + 2
    wheel = np.column_stack([data[k] for k in SENSOR_COLUMNS[9:13]])
    torques = np.column_stack([data[k] for k in SENSOR_COLUMNS[2:6]])
    gyro = data['yaw_rate_sensor_radps']
    x = initial_state(np.median(wheel[0]) * p[3], p)
    x[2] = gyro[0]
    x[5:] = wheel[0]
    P = np.diag([0.1, 0.3, 0.02, 0.01, 0.02, 1.0, 1.0, 1.0, 1.0]) ** 2
    R = np.diag([0.15, 0.15, 0.008, 0.6, 0.6, 0.6, 0.6]) ** 2
    Q = np.diag([0.1, 0.12, 0.03, 0.005, 0.03, 2.0, 2.0, 2.0, 2.0]) ** 2
    predictions = np.empty((len(data), 3))
    predictions[0] = x[:3]
    for i in range(1, len(data)):
        dt = data['time_s'][i] - data['time_s'][i - 1]
        chol = np.linalg.cholesky(positive_cov(P) * (n + lam))
        sigma = np.vstack([x, x + chol.T, x - chol.T])
        states, obs = sigma_predict(sigma, data['steering_rad'][i - 1], torques[i - 1], mu, p, dt)
        xm = weights @ states
        zm = weights @ obs
        xs = states - xm
        zs = obs - zm
        predP = xs.T * wc @ xs + Q * dt
        S = zs.T * wc @ zs + R
        cross = xs.T * wc @ zs
        gain = np.linalg.solve(positive_cov(S), cross.T).T
        measurement = np.r_[data['ax_mps2'][i], data['ay_mps2'][i], gyro[i], wheel[i]]
        x = xm + gain @ (measurement - zm)
        P = positive_cov(predP - gain @ S @ gain.T)
        if not np.isfinite(x).all():
            raise FloatingPointError(f'UKF diverged at row {i}')
        predictions[i] = x[:3]
    return predictions

class Chomp(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n else x

class CausalBlock(nn.Module):

    def __init__(self, cin, cout, dilation, dropout):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation), Chomp(pad), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(cout, cout, 3, padding=pad, dilation=dilation), Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)

class TCN(nn.Module):

    def __init__(self, n_features, hidden=64, levels=4, dropout=0.1):
        super().__init__()
        blocks = []
        cin = n_features
        for i in range(levels):
            blocks.append(CausalBlock(cin, hidden, 2 ** i, dropout))
            cin = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return self.head(self.body(x.transpose(1, 2)))

def condition_arrays(data_dir, out, p, mu, split_files):
    chunks = {}
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted({n for seq in split_files.values() for n in seq}):
        d = read_csv(data_dir / name, SENSOR_COLUMNS + REF_COLUMNS)
        key = hashlib.sha256((data_dir / name).read_bytes() + p.tobytes() + str(mu).encode() + b'ukf_v1').hexdigest()
        cached = out / (Path(name).stem + '_physics.npz')
        if cached.exists():
            z = np.load(cached)
            valid = str(z['key']) == key
        else:
            valid = False
        if valid:
            phy = z['physics']
        else:
            phy = ukf_branch(d, p, mu)
            np.savez_compressed(cached, key=key, physics=phy)
        features = np.column_stack([d[k] for k in SENSOR_COLUMNS[1:]] + [phy[:, k] for k in range(3)])
        ref = np.column_stack([d[k] for k in REF_COLUMNS])
        chunks[name] = {'features': features.astype(np.float32), 'physics': phy.astype(np.float32), 'target': ref.astype(np.float32), 'time': d['time_s']}
        print(f'Prepared {name}: {len(d)} samples', flush=True)
    return chunks

def windows(chunk, window):
    ends = np.arange(window - 1, len(chunk['features']))
    if len(ends) < 1:
        raise ValueError('Condition shorter than history window')
    X = np.stack([chunk['features'][i - window + 1:i + 1] for i in ends])
    return (X, chunk['physics'][ends], chunk['target'][ends], chunk['time'][ends])

def train_tcn(chunks, splits, out, epochs, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    W = STATE_CONFIG['window']
    allsets = {k: [windows(chunks[n], W) for n in seq] for k, seq in splits.items()}
    train = tuple((np.concatenate([v[j] for v in allsets['train']]) for j in range(4)))
    val = tuple((np.concatenate([v[j] for v in allsets['validation']]) for j in range(4)))
    feat_mean = train[0].reshape(-1, train[0].shape[-1]).mean(axis=0)
    feat_scale = np.maximum(train[0].reshape(-1, train[0].shape[-1]).std(axis=0), 0.001)

    def tensor_set(dataset):
        X, phy, y, _ = dataset
        return [torch.tensor((X - feat_mean) / feat_scale, dtype=torch.float32), torch.tensor(phy), torch.tensor(y)]
    xt, pt, yt = tensor_set(train)
    xv, pv, yv = tensor_set(val)
    scales = np.maximum(train[2].std(axis=0), [0.2, 0.1, 0.02]).astype(np.float32)
    residual_scale = np.maximum((train[2] - train[1]).std(axis=0), [0.05, 0.05, 0.01]).astype(np.float32)
    beta = train[2][:, 1] / np.maximum(train[2][:, 0], 0.5)
    beta_scale = max(float(beta.std()), 0.01)
    center = np.zeros(3, np.float32)
    output_scale = residual_scale
    net = TCN(len(feat_mean))
    opt = torch.optim.AdamW(net.parameters(), lr=STATE_CONFIG['learning_rate'], weight_decay=0.0001)
    sc = torch.tensor(scales)
    outs = torch.tensor(output_scale)
    ctr = torch.tensor(center)
    best = None
    best_val = np.inf
    history = []
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(xt))
        losses = []
        for start in range(0, len(order), STATE_CONFIG['batch_size']):
            ids = order[start:start + STATE_CONFIG['batch_size']]
            raw = net(xt[ids])
            delta = raw * outs + ctr
            pred = pt[ids] + delta
            state_loss = (((pred - yt[ids]) / sc) ** 2).mean()
            beta_pred = torch.atan2(pred[:, 1], pred[:, 0].clamp(min=0.5))
            beta_ref = torch.atan2(yt[ids, 1], yt[ids, 0].clamp(min=0.5))
            loss = state_loss + STATE_CONFIG['lambda_beta'] * (((beta_pred - beta_ref) / beta_scale) ** 2).mean()
            loss += STATE_CONFIG['lambda_residual'] * (raw ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            vp = net(xv) * outs + ctr
            vp = vp + pv
            value = float((((vp - yv) / sc) ** 2).mean())
        history.append([epoch + 1, float(np.mean(losses)), value])
        if value < best_val:
            best_val = value
            best = copy.deepcopy(net.state_dict())
            best_epoch = epoch + 1
        if (epoch + 1) % 5 == 0:
            print(f"{'DeepPhy'} epoch {epoch + 1}: validation loss={value:.5f}", flush=True)
    net.load_state_dict(best)
    checkpoint = {'network': best, 'feature_mean': torch.tensor(feat_mean), 'feature_scale': torch.tensor(feat_scale), 'output_scale': torch.tensor(output_scale), 'center': torch.tensor(center), 'config': STATE_CONFIG, 'seed': seed, 'best_epoch': best_epoch, 'best_validation_loss': best_val, 'baseline': False, 'training_files': splits['train'], 'validation_files': splits['validation'], 'epochs_requested': epochs, 'scope': 'new reference implementation; data kind in manifest, not original manuscript checkpoint'}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out / 'deepphy.pt')
    save_csv(out / 'training.csv', np.asarray(history), ['epoch', 'train_loss', 'validation_loss'])
    return checkpoint

def infer_state(checkpoint, X, phy):
    cfg = checkpoint['config']
    net = TCN(len(checkpoint['feature_mean']), cfg['hidden_channels'], cfg['levels'], cfg['dropout'])
    net.load_state_dict(checkpoint['network'])
    net.eval()
    if checkpoint['baseline']:
        X = X[:, :, :-3]
    X = (torch.tensor(X) - checkpoint['feature_mean']) / checkpoint['feature_scale']
    values = []
    with torch.no_grad():
        for j in range(0, len(X), 256):
            values.append((net(X[j:j + 256]) * checkpoint['output_scale'] + checkpoint['center']).numpy())
    pred = np.concatenate(values)
    return pred if checkpoint['baseline'] else phy + pred

def append_beta(state):
    return np.column_stack([state[:, 0], state[:, 1], np.arctan2(state[:, 1], np.maximum(state[:, 0], 0.5)), state[:, 2]])

def verify_checkpoint_plant(checkpoint, p, mu):
    if 'physical_parameters' not in checkpoint:
        raise ValueError('Checkpoint lacks physical-parameter provenance; retrain or use the supplied documented checkpoint')
    if not np.allclose(checkpoint['physical_parameters'].numpy(), p, rtol=0, atol=1e-12) or abs(checkpoint['friction'] - mu) > 1e-12:
        raise ValueError('Physical parameters differ from training; supply the matching --params file')
