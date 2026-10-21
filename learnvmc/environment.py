"""Gymnasium DLC environment, physical QP allocator and optional CarSim bridge."""
from __future__ import annotations
import sys
import math
from pathlib import Path
from datetime import timedelta
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
from realcalib.vehicle_model import load_params, initial_state, dynamics, propagate

ENV_CONFIG = {'dt': 0.05, 'duration': 12.0, 'max_moment_nm': 3500.0, 'max_torque_nm': 1000.0, 'reward_Q': [100.0, 200.0, 100.0], 'reward_R': 1e-07, 'violation_penalty': 50.0, 'integration_step': 0.001, 'reference_Cf': 160000.0, 'reference_Cr': 160000.0, 'steering_feedback_gain': 0.8, 'steering_limit_rad': 0.3, 'steering_rate_radps': 0.7}
OBS_NAMES = ['steering_rad', 'vx_mps', 'heading_error_rad', 'beta_rad', 'yaw_rate_radps', 'beta_error_rad', 'yaw_rate_error_radps']

OBS_SCALE = np.array([0.15, 20.0, 0.3, 0.15, 0.6, 0.15, 0.6], np.float32)

@njit(cache=False)
def path_reference(X, geom):
    dx1, dx2, dy1, dy2, x1, x2 = geom
    z1 = 2.4 / dx1 * (X - x1) - 1.2
    z2 = 2.4 / dx2 * (X - x2) - 1.2
    th1, th2 = (math.tanh(z1), math.tanh(z2))
    Y = 0.5 * dy1 * (1 + th1) - 0.5 * dy2 * (1 + th2)
    slope = 0.5 * dy1 * (2.4 / dx1) * (1 - th1 * th1) - 0.5 * dy2 * (2.4 / dx2) * (1 - th2 * th2)
    second = -dy1 * (2.4 / dx1) ** 2 * th1 * (1 - th1 * th1) + dy2 * (2.4 / dx2) ** 2 * th2 * (1 - th2 * th2)
    return (Y, math.atan(slope), second / (1 + slope * slope) ** 1.5)

@njit(cache=False)
def allocate_torque(total, moment, fz, mu, p, tmax=1000.0):
    """Exact diagonal QP with two equal-track equalities and box bounds.

    Minimize sum((T/(rw*mu*Fz))**2). The two equalities uniquely fix
    left/right totals, leaving two independent bounded convex splits. Requests
    are projected to feasible side totals first; achieved moment is returned.
    """
    rw = p[3]
    track = 0.5 * (p[4] + p[5])
    caps = np.minimum(np.full(4, tmax), mu * fz * rw)
    total = min(caps.sum(), max(-caps.sum(), total))
    lcap = caps[0] + caps[2]
    rcap = caps[1] + caps[3]
    right = min(rcap, max(-rcap, 0.5 * total + moment * rw / track))
    right = min(min(rcap, total + lcap), max(max(-rcap, total - lcap), right))
    left = total - right
    torque = np.empty(4)
    for side in range(2):
        a = side
        b = side + 2
        target = left if side == 0 else right
        wa = 1 / max(fz[a] * mu * rw, 1e-06) ** 2
        wb = 1 / max(fz[b] * mu * rw, 1e-06) ** 2
        first = wb / (wa + wb) * target
        torque[a] = min(min(caps[a], target + caps[b]), max(max(-caps[a], target - caps[b]), first))
        torque[b] = target - torque[a]
    achieved = track / (2 * rw) * (-torque[0] + torque[1] - torque[2] + torque[3])
    return (torque, achieved)

def desired_states(delta, vx, p, mu):
    m = p[0]
    lf, lr = p[6:8]
    L = lf + lr
    cf = ENV_CONFIG['reference_Cf']
    cr = ENV_CONFIG['reference_Cr']
    K = m / L ** 2 * (lr / cf - lf / cr)
    den = max(0.2, 1 + K * vx ** 2)
    yaw = vx / L / den * delta
    yaw = float(np.clip(yaw, -mu * p[2] / max(abs(vx), 0.5), mu * p[2] / max(abs(vx), 0.5)))
    beta = (lr - m * lf * vx ** 2 / (L * cr)) / (den * L) * delta
    return (float(beta), yaw)

class CarSimBackend:
    """Bridge to the author's CarsimManager. Channel names/order/units are explicit.

    A new manager is created on reset because its native reset is unimplemented.
    The factory parameter also permits interface tests without a CarSim licence.
    """
    def __init__(self, config, factory=None):
        self.config = config
        self.factory = factory
        self.manager = None

    def reset(self, dt):
        self.close()
        cfg = self.config
        factory = self.factory
        if factory is None:
            if cfg.get('pycarsimlib_path'):
                sys.path.insert(0, str(Path(cfg['pycarsimlib_path']).resolve()))
            from pycarsimlib.core import CarsimManager
            factory = CarsimManager
        self.manager = factory(cfg['database_dir'], 'normal_vehicle')
        try:
            imports = cfg['torque_imports'] + [cfg['steering_import']]
            exports = [item[0] for item in cfg['signals'].values()]
            if len(set(imports)) != len(imports) or len(set(exports)) != len(exports):
                raise ValueError('CarSim channel names must be unique and in project order')
            if self.manager.num_of_import_values != len(imports) or self.manager.num_of_export_values != len(exports):
                raise ValueError('CarSim IMPORT/EXPORT count differs from the bridge JSON')
            ratio = dt / self.manager.t_step
            if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0):
                raise ValueError('Controller dt must be an integer multiple of the CarSim solver step')
            self.manager.import_label_array = imports
            self.manager.export_label_array = exports
            self.dt = dt
            self.start_time = self.manager.t_current
            raw = dict(zip(exports, self.manager.export_array))
            return self._frame(raw), False
        except Exception:
            self.close()
            raise

    def _frame(self, raw):
        values = {key: float(raw[name]) * scale for key, (name, scale) in self.config['signals'].items()}
        required = ['vx','vy','yaw_rate','roll','roll_rate','X','Y','yaw','fz_fl','fz_fr','fz_rl','fz_rr']
        if set(values) != set(required) or not np.isfinite(list(values.values())).all():
            raise ValueError('CarSim must provide all 12 finite SI signals in the example JSON')
        fz = np.array([values['fz_'+w] for w in ['fl','fr','rl','rr']])
        if np.any(fz <= 0):
            raise ValueError('CarSim vertical loads must be positive N')
        # Only these five body states are required; no wheel measurements are invented.
        body = np.array([values[k] for k in required[:5]])
        pose = np.array([values[k] for k in ['X','Y','yaw']])
        return body, pose, fz

    def step(self, steering_rad, torque):
        if self.manager is None:
            raise RuntimeError('Reset the CarSim backend before stepping')
        commands = dict(zip(self.config['torque_imports'], map(float, torque)))
        commands[self.config['steering_import']] = float(steering_rad) * self.config['steering_scale']
        previous_time = self.manager.t_current
        raw, status, timestamp = self.manager.step(commands, timedelta(seconds=self.dt))
        if not np.isfinite(timestamp) or not np.isclose(timestamp-previous_time,self.dt,atol=1e-7,rtol=0):
            raise RuntimeError('CarSim did not advance by exactly one controller step')
        return self._frame(raw), bool(status)

    def close(self):
        if self.manager is not None:
            manager, self.manager = self.manager, None
            manager.close()


class DLCEnv(gym.Env):
    """Gymnasium environment: normalized 7-state observation, scalar yaw moment.

    TD3 -> moment -> bounded QP -> four wheel torques -> 8DOF / CarSim.
    Path following and speed regulation are common external inputs.
    """
    metadata = {'render_modes': [], 'render_fps': 20}

    def __init__(self, randomize=True, params=None, backend=None):
        super().__init__()
        if backend is not None and randomize:
            raise ValueError('Configure CarSim vehicle randomization in its project; use randomize=False')
        self.randomize, self.backend = randomize, backend
        self.base_p, self.base_mu, self.act = load_params(params)
        self.dt = ENV_CONFIG['dt']
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        scenario = (options or {}).get('scenario')
        r = self.rng
        self.p = self.base_p.copy()
        if scenario is not None:
            self.speed, self.mu = float(scenario['speed_mps']), float(scenario['mu'])
            self.geom = np.array(scenario['geometry'], float)
            factor = float(scenario.get('mass_factor', 1.0))
            if self.backend is not None and factor != 1:
                raise ValueError('CarSim mass must be configured in the external project')
        elif self.randomize:
            self.speed, self.mu, factor = r.uniform(30,75)/3.6, r.uniform(.4,.9), r.uniform(.9,1.1)
            self.geom = np.r_[r.uniform(20,30,2), r.uniform(3,6,2), 20.,60.]
        else:
            self.speed, self.mu, factor = 60/3.6, .85, 1.
            self.geom = np.array([25.,25.,4.05,4.05,20.,60.])
        self.p[:2] *= factor
        self.x = initial_state(self.speed, self.p)
        self.pose = np.array([0., path_reference(0.,self.geom)[0], 0.])
        if self.randomize and scenario is None:
            self.pose[1] += r.uniform(-.15,.15)
            self.pose[2] += r.uniform(-.015,.015)
        self.delta = self.actual_delta = self.prev_moment = self.elapsed = 0.
        self.steps = 0
        self.last_torque = np.zeros(4)
        self.done = self.backend_done = False
        if self.backend is not None:
            frame, self.backend_done = self.backend.reset(self.dt)
            self.x, self.pose, self.fz = frame
        self._steer()
        return self.observation(), {}

    def _steer(self):
        X, Y, psi = self.pose
        ref, heading, curvature = path_reference(X,self.geom)
        heading_error = (psi-heading+np.pi)%(2*np.pi)-np.pi
        target = (self.p[6]+self.p[7])*curvature-heading_error-np.arctan(
            ENV_CONFIG['steering_feedback_gain']*(Y-ref)/max(self.x[0],2.))
        target = np.clip(target,-ENV_CONFIG['steering_limit_rad'],ENV_CONFIG['steering_limit_rad'])
        limit = ENV_CONFIG['steering_rate_radps']*self.dt
        self.delta += float(np.clip(target-self.delta,-limit,limit))

    def observation(self):
        heading = path_reference(self.pose[0],self.geom)[1]
        error = (self.pose[2]-heading+np.pi)%(2*np.pi)-np.pi
        beta = np.arctan2(self.x[1],max(self.x[0],.5))
        bd, rd = desired_states(self.delta,self.x[0],self.p,self.mu)
        raw = np.array([self.delta,self.x[0],error,beta,self.x[2],bd-beta,rd-self.x[2]],np.float32)
        return raw/OBS_SCALE

    def control_context(self):
        fz = self.fz if self.backend is not None else dynamics(
            self.x,self.actual_delta,self.last_torque,self.mu,self.p)[3]
        drag = .5*self.p[12]*self.p[11]*self.p[10]*self.x[0]**2
        total = (self.p[0]*.8*(self.speed-self.x[0])+drag+self.p[0]*self.p[2]*self.p[9])*self.p[3]
        return float(total), fz.copy()

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset before stepping or after an episode ends')
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Action must be one finite normalized yaw-moment command')
        requested = float(np.clip(action[0],-1,1))*ENV_CONFIG['max_moment_nm']
        total, fz = self.control_context()
        torque, achieved = allocate_torque(total,requested,fz,self.mu,self.p,ENV_CONFIG['max_torque_nm'])
        if self.backend is None:
            self.x = propagate(self.x,self.actual_delta,torque,self.mu,self.p,self.dt,ENV_CONFIG['integration_step'])
            tau = self.act.get('steering_tau',.03)
            self.actual_delta += (1-np.exp(-self.dt/max(tau,1e-5)))*(
                self.act.get('steering_scale',1.)*self.delta+self.act.get('steering_bias',0.)-self.actual_delta)
            vx,vy,r = self.x[:3]
            psi = self.pose[2]
            self.pose += self.dt*np.array([vx*np.cos(psi)-vy*np.sin(psi),vx*np.sin(psi)+vy*np.cos(psi),r])
        else:
            frame, self.backend_done = self.backend.step(self.delta,torque)
            self.x,self.pose,self.fz = frame
        self.steps += 1
        self.elapsed = self.steps*self.dt
        self.last_torque = torque
        vx,vy,r = self.x[:3]
        ref,heading,_ = path_reference(self.pose[0],self.geom)
        ey = self.pose[1]-ref
        beta = np.arctan2(vy,max(vx,.5))
        bd,rd = desired_states(self.delta,vx,self.p,self.mu)
        epsi = (self.pose[2]-heading+np.pi)%(2*np.pi)-np.pi
        violation = abs(beta)>.25 or abs(rd-r)>.6 or abs(self.x[3])>.25
        errors = np.array([epsi,bd-beta,rd-r])
        cost = np.dot(ENV_CONFIG['reward_Q'],errors**2)+ENV_CONFIG['reward_R']*(requested-self.prev_moment)**2
        cost += ENV_CONFIG['violation_penalty']*violation
        terminated = bool(self.backend_done or not np.isfinite(self.x).all() or abs(ey)>8 or abs(beta)>.6 or vx<2 or abs(self.x[3])>.5)
        truncated = bool(self.elapsed >= ENV_CONFIG['duration']-1e-8)
        if terminated:
            cost += 100
        self.prev_moment, self.done = requested, terminated or truncated
        info = dict(time_s=self.elapsed,X_m=float(self.pose[0]),Y_m=float(self.pose[1]),Y_ref_m=float(ref),
                    beta_rad=float(beta),beta_ref_rad=float(bd),yaw_rate_radps=float(r),yaw_ref_radps=float(rd),
                    lateral_error_m=float(ey),requested_moment_nm=requested,achieved_moment_nm=float(achieved),
                    **{f'torque_{w}_nm':float(torque[i]) for i,w in enumerate(['fl','fr','rl','rr'])})
        self._steer()
        return self.observation(), -float(cost), terminated, truncated, info

    def close(self):
        if self.backend is not None:
            self.backend.close()
