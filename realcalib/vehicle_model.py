"""Eight-DOF body/wheel dynamics and open-loop input/output simulation (SI).

Eight mechanical DOFs require nine first-order states: vx, vy, yaw rate,
roll angle/rate, and four wheel angular speeds. Global pose is external.
Steering ratio is 21; steering_rad inputs are already road-wheel radians.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from numba import njit
STATE_NAMES = ['vx', 'vy', 'yaw_rate', 'roll', 'roll_rate', 'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr']

PARAM_NAMES = ['m', 'ms', 'g', 'rw', 'bsf', 'bsr', 'lf', 'lr', 'hs', 'rolling_resistance', 'air_density', 'frontal_area', 'drag_coefficient', 'ix', 'iz', 'jw', 'k_phi_front', 'k_phi_rear', 'c_phi', 'gain_fx', 'gain_fy']

NOMINAL = np.array([1737.0, 1617.0, 9.8, 0.308, 1.48, 1.485, 1.345, 1.385, 0.3, 0.015, 1.206, 1.6, 0.3, 288.0, 2712.0, 0.9, 14381.0, 22002.0, 1000.0, 1.0, 2.6])

STEERING_RATIO = 21.0

def steering_wheel_to_road(values, unit='deg', sign=1.0):
    values = np.asarray(values, float)
    if unit == 'deg':
        values = np.deg2rad(values)
    elif unit != 'rad':
        raise ValueError('Steering-wheel angle unit must be deg or rad')
    return sign * values / STEERING_RATIO

@njit(cache=False)
def signed_floor(x, floor=0.5):
    return max(x, floor) if x >= 0 else min(x, -floor)

@njit(cache=False)
def dynamics(x, steering, torque, mu, p):
    """Body-frame SI equations; returns dx, body ax/ay, wheel vertical loads.

    Retains the source model's combined-slip Magic Formula, three load-transfer
    iterations, friction-circle guard and derivative guards. No reference
    trajectory or measured target output is used by this plant.
    """
    m, ms, g, rw, tf, tr, lf, lr, h = p[:9]
    fr, rho, area, cd, ix, iz, jw, kf, kr, damping, gx, gy = p[9:]
    vx, vy, r, phi, phid = x[:5]
    c, s = (math.cos(steering), math.sin(steering))
    v = signed_floor(vx)
    alpha = np.array([-(vy + lf * r) / signed_floor(v - 0.5 * tf * r) + steering, -(vy + lf * r) / signed_floor(v + 0.5 * tf * r) + steering, -(vy - lr * r) / signed_floor(v - 0.5 * tr * r), -(vy - lr * r) / signed_floor(v + 0.5 * tr * r)])
    alpha = np.minimum(np.maximum(alpha, -0.55), 0.55)
    vl = np.array([(v - 0.5 * tf * r) * c + (vy + lf * r) * s, (v + 0.5 * tf * r) * c + (vy + lf * r) * s, v - 0.5 * tr * r, v + 0.5 * tr * r])
    slip = np.empty(4)
    for j in range(4):
        slip[j] = min(1.0, max(-1.0, (rw * x[5 + j] - vl[j]) / max(abs(vl[j]), abs(rw * x[5 + j]), 0.5)))
    ax, ay = (0.0, 0.0)
    fz, fx, fy = (np.zeros(4), np.zeros(4), np.zeros(4))
    deriv = np.zeros(9)
    limit = max(3.0, 1.5 * max(abs(mu), 0.2) * g)
    for _ in range(3):
        roll_f = (kf * phi + damping * phid) / tf
        roll_r = (kr * phi + damping * phid) / tr
        length = lf + lr
        fz[0] = m * g * lr / (2 * length) - m * ax * h / (2 * length) - m * ay * h * lr / (tf * length) - roll_f
        fz[1] = m * g * lr / (2 * length) - m * ax * h / (2 * length) + m * ay * h * lr / (tf * length) + roll_f
        fz[2] = m * g * lf / (2 * length) + m * ax * h / (2 * length) - m * ay * h * lf / (tr * length) - roll_r
        fz[3] = m * g * lf / (2 * length) + m * ax * h / (2 * length) + m * ay * h * lf / (tr * length) + roll_r
        for j in range(4):
            fz[j] = min(0.8 * m * g, max(50.0, fz[j]))
            z = fz[j] / 1000.0
            sx = slip[j] / max(1.0 + slip[j], 0.05)
            sy = math.tan(alpha[j]) / max(1.0 + slip[j], 0.05)
            sig = math.sqrt(sx * sx + sy * sy)
            dx = signed_floor(-21.3 * z * z + 1144.0 * z, 1e-06)
            bx = (49.6 * z * z + 226.0 * z) / math.exp(0.069 * z) / (1.65 * dx)
            ex = -0.006 * z * z + 0.056 * z + 0.486
            fx0 = mu * dx * math.sin(1.65 * math.atan(bx * sig * 100 - ex * (bx * sig * 100 - math.atan(bx * sig * 100))))
            dy = signed_floor(-22.1 * z * z + 1011.0 * z, 1e-06)
            by = 1078.0 * math.sin(1.82 * math.atan(0.208 * z)) / (1.3 * dy)
            ey = -0.354 * z + 0.707
            sd = sig * 180 / math.pi
            fy0 = mu * dy * math.sin((5 - mu) / 4 * 1.3 * math.atan((2 - mu) * by * sd * (1 - ey) + ey * math.atan((2 - mu) * by * sd)))
            fx[j] = gx * sx / (sig + 1e-10) * fx0
            fy[j] = gy * sy / (sig + 1e-10) * fy0
            scale = min(1.0, max(abs(mu), 0.05) * fz[j] * 1.1 / max(math.hypot(fx[j], fy[j]), 1e-09))
            fx[j] *= scale
            fy[j] *= scale
        moment = 0.5 * tf * (fx[1] * c - fy[1] * s - (fx[0] * c - fy[0] * s)) + 0.5 * tr * (fx[3] - fx[2]) + lf * ((fy[0] + fy[1]) * c + (fx[0] + fx[1]) * s) - lr * (fy[2] + fy[3])
        lateral = (fy[0] + fy[1]) * c + fy[2] + fy[3] + (fx[0] + fx[1]) * s
        aero = 0.5 * cd * area * rho * vx * abs(vx)
        rolling = m * g * fr * math.tanh(vx / 0.5)
        axraw = ((fx[0] + fx[1]) * c + fx[2] + fx[3] - (fy[0] + fy[1]) * s - aero - rolling - ms * h * (moment / iz) * phi) / m
        deriv[0] = min(limit, max(-limit, axraw + r * vy))
        deriv[1] = min(limit, max(-limit, (lateral - ms * h / ix * (ms * h * g * phi - 2 * damping * phid - (kf + kr) * phi)) / (m + ms * ms * h * h / ix) - r * vx))
        deriv[2] = moment / iz
        deriv[3] = min(3.0, max(-3.0, phid))
        deriv[4] = min(20.0, max(-20.0, (ms * h * g * phi - 2 * damping * phid - (kf + kr) * phi + ms * h / m * lateral) / (ix + ms * ms * h * h / m)))
        for j in range(4):
            deriv[5 + j] = min(5000.0, max(-5000.0, (torque[j] - fx[j] * rw - fz[j] * rw * fr) / jw))
        ax = deriv[0] - r * vy
        ay = deriv[1] + r * vx
    return (deriv, ax, ay, fz)

@njit(cache=False)
def propagate(x, steering, torque, mu, p, dt, max_step=0.001):
    count = max(1, int(math.ceil(dt / max_step)))
    h = dt / count
    state = x.copy()
    for _ in range(count):
        k1 = dynamics(state, steering, torque, mu, p)[0]
        k2 = dynamics(state + 0.5 * h * k1, steering, torque, mu, p)[0]
        state += h * k2
    return state

def initial_state(speed, p=NOMINAL):
    return np.r_[float(speed), 0.0, 0.0, 0.0, 0.0, np.full(4, float(speed) / p[3])]

def load_params(path):
    if path is None:
        return (NOMINAL.copy(), 0.8, {'steering_scale': 1.0, 'steering_bias': 0.0, 'steering_tau': 0.03})
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    vals = payload['physical_parameters']
    p = np.array([float(vals[n]) for n in PARAM_NAMES])
    return (p, float(payload['friction']), payload.get('actuator', {}))

def scores(y, pred, names):
    out = {}
    for j, name in enumerate(names):
        err = pred[:, j] - y[:, j]
        denom = float(np.sum((y[:, j] - y[:, j].mean()) ** 2))
        out[name] = {'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt(np.mean(err ** 2))), 'max_abs': float(np.abs(err).max()), 'r2': float(1 - np.sum(err ** 2) / denom) if denom > 1e-12 else None}
    return out

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def read_csv(path, required):
    a = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig')
    if a.ndim != 1 or len(a) < 3 or any((k not in a.dtype.names for k in required)):
        raise ValueError('CSV requires >=3 rows and columns: ' + ', '.join(required))
    for k in required:
        if not np.isfinite(a[k]).all():
            raise ValueError('Non-finite column: ' + k)
    if np.any(np.diff(a['time_s']) <= 0):
        raise ValueError('Timestamps must strictly increase')
    return a

def save_csv(path, values, columns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter=',', header=','.join(columns), comments='', fmt='%.10g')

def plot_series(path, t, reference, pred, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.5 * len(names)), sharex=True, squeeze=False)
    for j, name in enumerate(names):
        axes[j, 0].plot(t, reference[:, j], label='Reference', lw=1)
        axes[j, 0].plot(t, pred[:, j], label='Estimate', lw=1)
        axes[j, 0].set_ylabel(name)
        axes[j, 0].grid(alpha=0.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

CAL_COLUMNS = ['time_s', 'steering_rad', 'torque_fl_nm', 'torque_fr_nm', 'torque_rl_nm', 'torque_rr_nm', 'vx_mps', 'ay_mps2', 'yaw_rate_radps']

@njit(cache=False)
def rollout_cal(t, steer, torques, x0, p, mu, scale, bias, tau):
    x = x0.copy()
    delta = scale * steer[0] + bias
    states = np.empty((len(t), 9))
    outputs = np.empty((len(t), 3))
    for i in range(len(t)):
        states[i] = x
        _, ax, ay, _ = dynamics(x, delta, torques[i], mu, p)
        outputs[i] = np.array([x[0], ay, x[2]])
        if i + 1 < len(t):
            dt = t[i + 1] - t[i]
            next_delta = delta + (1 - math.exp(-dt / max(tau, 1e-05))) * (scale * steer[i] + bias - delta)
            x = propagate(x, delta, torques[i], mu, p, dt)
            delta = next_delta
    return (outputs, states)

def predict_cal(data, p, mu, act):
    t = data['time_s']
    torque = np.column_stack([data[n] for n in CAL_COLUMNS[2:6]])
    x = initial_state(data['vx_mps'][0], p)
    x[2] = data['yaw_rate_radps'][0]
    if 'vy_mps' in data.dtype.names:
        x[1] = data['vy_mps'][0]
    return rollout_cal(t, data['steering_rad'], torque, x, p, mu, act.get('steering_scale', 1.0), act.get('steering_bias', 0.0), act.get('steering_tau', 0.03))

def cal_target(data):
    return np.column_stack([data[n] for n in CAL_COLUMNS[-3:]])
