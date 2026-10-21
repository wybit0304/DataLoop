"""Train -> save -> evaluate -> export -> deploy the hierarchical TD3 policy."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from realcalib.vehicle_model import load_params, read_csv, save_csv, write_json
from learnvmc.environment import DLCEnv, CarSimBackend, ENV_CONFIG, OBS_NAMES, OBS_SCALE, allocate_torque
from learnvmc.td3 import TD3, Replay, TD3_CONFIG

HERE = Path(__file__).resolve().parent
WHEELS = ['fl','fr','rl','rr']
DEPLOY_COLUMNS = ['time_s'] + OBS_NAMES + ['total_torque_nm','mu'] + ['fz_'+w+'_n' for w in WHEELS]


def make_env(args, randomize=False):
    backend = None
    if args.carsim_config:
        cfg = json.loads(args.carsim_config.read_text(encoding='utf-8'))
        backend = CarSimBackend(cfg)
    return DLCEnv(randomize=randomize and backend is None, params=args.params, backend=backend)


def scenarios(seed, count):
    rng = np.random.default_rng(seed)
    return [dict(speed_mps=float(rng.uniform(40,70)/3.6), mu=float(rng.uniform(.55,.9)),
                 mass_factor=float(rng.uniform(.9,1.1)),
                 geometry=np.r_[rng.uniform(20,30,2),rng.uniform(3,6,2),20.,60.].tolist()) for _ in range(count)]


def rollout(env, agent=None, scenario=None):
    obs, _ = env.reset(options={'scenario':scenario})
    rows, deployment, total_return = [], [], 0.
    while True:
        total, fz = env.control_context()
        deployment.append(np.r_[env.elapsed,obs*OBS_SCALE,total,env.mu,fz])
        action = agent.act(obs) if agent is not None else np.zeros(1,np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(info)
        total_return += reward
        if terminated or truncated:
            break
    data = lambda key: np.array([row[key] for row in rows])
    beta_error = data('beta_rad')-data('beta_ref_rad')
    yaw_error = data('yaw_rate_radps')-data('yaw_ref_radps')
    metric = dict(episode_return=total_return, steps=len(rows), terminated=terminated,
                  beta_mae_rad=float(np.abs(beta_error).mean()),yaw_mae_radps=float(np.abs(yaw_error).mean()),
                  lateral_rmse_m=float(np.sqrt(np.mean(data('lateral_error_m')**2))),
                  integrated_error=float(np.sum(beta_error**2+yaw_error**2+data('lateral_error_m')**2)*env.dt))
    return metric, rows, np.asarray(deployment)


def evaluate(args, agent):
    env = make_env(args)
    # A CarSim run uses its project's initial state and vehicle configuration.
    cases = [None] if args.carsim_config else scenarios(200001,8)
    records = {key:[] for key in ['ADS','TD3']}
    try:
        for label, policy in [('ADS',None),('TD3',agent)]:
            for i, case in enumerate(cases):
                metric, rows, deployment = rollout(env,policy,case)
                records[label].append(metric)
                if i == 0:
                    keys = list(rows[0])
                    save_csv(args.out/(label+'_trajectory.csv'),np.array([[r[k] for k in keys] for r in rows]),keys)
                    if label == 'TD3':
                        save_csv(args.out/'deployment_inputs.csv',deployment,DEPLOY_COLUMNS)
    finally:
        env.close()
    for baseline, learned in zip(records['ADS'],records['TD3']):
        learned['Q'] = None if baseline['terminated'] or learned['terminated'] else 2-learned['integrated_error']/max(baseline['integrated_error'],1e-12)
    summary = {}
    for label, results in records.items():
        summary[label] = {key:float(np.mean([r[key] for r in results])) for key in
                          ['episode_return','beta_mae_rad','yaw_mae_radps','lateral_rmse_m','integrated_error']}
        summary[label]['failures'] = sum(r['terminated'] for r in results)
    q = [r['Q'] for r in records['TD3'] if r['Q'] is not None]
    summary['TD3']['Q_mean'] = float(np.mean(q)) if q else None
    result = dict(backend='CarSim' if args.carsim_config else '8DOF',summary=summary,cases=cases,per_case=records)
    write_json(args.out/'evaluation.json',result)
    print(json.dumps(summary,indent=2))


def train(args, agent):
    env = make_env(args,randomize=True)
    env.reset(seed=args.seed)
    obs, _ = env.reset()
    rng = np.random.default_rng(args.seed)
    replay = Replay(min(args.steps+1,TD3_CONFIG['buffer_size']),args.seed)
    best_value = -np.inf
    history, validation = [], []
    episode_return, episode_steps = 0., 0
    checkpoint = args.out/'td3.pt'
    plant = dict(physical_parameters=env.base_p.tolist(),friction=env.base_mu,actuator=env.act)
    try:
        for step in range(1,args.steps+1):
            if step <= 2000:
                action = rng.uniform(-.25,.25,1).astype(np.float32)
            else:
                noise = TD3_CONFIG['exploration_std']*max(.15,np.exp(-step/50000))
                action = np.clip(agent.act(obs)+rng.normal(0,noise,1),-1,1).astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.add(obs,action,reward*TD3_CONFIG['replay_reward_scale'],next_obs,terminated)
            obs = next_obs
            if step >= 2000 and replay.size >= TD3_CONFIG['batch_size']:
                agent.update(replay)
            episode_return += reward
            episode_steps += 1
            if terminated or truncated:
                history.append([step,episode_return,episode_steps])
                obs, _ = env.reset()
                episode_return, episode_steps = 0., 0
            if step%5000 == 0 or step == args.steps:
                cases = [None] if args.carsim_config else scenarios(100001,4)
                value = float(np.mean([rollout(env,agent,case)[0]['episode_return'] for case in cases]))
                if value > best_value:
                    best_value = value
                    agent.save(checkpoint,env_config=ENV_CONFIG,plant=plant,seed=args.seed,
                               selected_step=step,validation_return=value,data_kind='synthetic DLC' if not args.carsim_config else 'CarSim project')
                validation.append([step,value,best_value])
                print(f'TD3 {step}/{args.steps}: validation={value:.3f}, best={best_value:.3f}',flush=True)
                obs, _ = env.reset()
                episode_return, episode_steps = 0., 0
    finally:
        env.close()
    save_csv(args.out/'training.csv',np.asarray(history).reshape(-1,3),['step','episode_return','episode_steps'])
    save_csv(args.out/'validation.csv',np.asarray(validation),['step','episode_return','best_return'])
    agent.load(checkpoint)
    print('Saved selected network, critics, targets and optimizers:',checkpoint)


def export(args, agent, metadata):
    target = args.out/'policy.ts'
    agent.export(target,OBS_SCALE,ENV_CONFIG['max_moment_nm'])
    with target.open('rb') as stream:
        loaded = torch.jit.load(stream).eval()
    raw = torch.tensor([[.02,15.,.01,.005,.1,-.002,.02]],dtype=torch.float32)
    with torch.no_grad():
        expected = agent.actor(raw/torch.tensor(OBS_SCALE))*ENV_CONFIG['max_moment_nm']
        torch.testing.assert_close(loaded(raw),expected)
    write_json(args.out/'policy.json',dict(input_columns=OBS_NAMES,output='requested yaw moment (Nm)',
        observation_scale=OBS_SCALE.tolist(),max_moment_nm=ENV_CONFIG['max_moment_nm'],
        steering_ratio=21,plant=metadata['plant'],lower_layer='environment.allocate_torque'))
    print('Exported and reloaded TorchScript policy:',target)


def deploy(args):
    """Offline deployment pipeline: SI observations -> actor -> QP -> torque CSV."""
    with (args.out/'policy.ts').open('rb') as stream:
        model = torch.jit.load(stream).eval()
    meta = json.loads((args.out/'policy.json').read_text(encoding='utf-8'))
    source = args.input or args.out/'deployment_inputs.csv'
    data = read_csv(source,DEPLOY_COLUMNS)
    states = np.column_stack([data[k] for k in OBS_NAMES]).astype(np.float32)
    p = np.array(meta['plant']['physical_parameters'])
    with torch.no_grad():
        moments = model(torch.from_numpy(states)).numpy().reshape(-1)
    rows = []
    for i, moment in enumerate(moments):
        fz = np.array([data['fz_'+w+'_n'][i] for w in WHEELS])
        if data['mu'][i] <= 0 or np.any(fz <= 0):
            raise ValueError('Deployment requires positive friction and vertical loads')
        torques, achieved = allocate_torque(data['total_torque_nm'][i],float(moment),fz,data['mu'][i],p,ENV_CONFIG['max_torque_nm'])
        rows.append(np.r_[data['time_s'][i],moment,achieved,torques])
    save_csv(args.out/'deployed_torques.csv',np.asarray(rows),
             ['time_s','requested_moment_nm','achieved_moment_nm']+['torque_'+w+'_nm' for w in WHEELS])
    print('Saved actor + QP deployment outputs:',args.out/'deployed_torques.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate','export','deploy'],nargs='?',default='evaluate')
    parser.add_argument('--steps',type=int,default=30000)
    parser.add_argument('--seed',type=int,default=2)
    parser.add_argument('--params',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=HERE/'outputs')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--carsim-config',type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.finfo(np.float32); np.finfo(np.float64)
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint is None:
        args.checkpoint = args.out/'td3.pt'
    if args.mode == 'deploy':
        deploy(args)
        return
    agent = TD3()
    if args.mode == 'train':
        train(args,agent)
        evaluate(args,agent)
        return
    if not args.checkpoint.is_file():
        parser.error('No trained checkpoint found. Run train first with the same --out, or supply --checkpoint.')
    metadata = agent.load(args.checkpoint)
    if metadata.get('env_config',ENV_CONFIG) != ENV_CONFIG:
        raise ValueError('Environment configuration differs from the checkpoint')
    p,mu,act = load_params(args.params)
    if metadata['plant'] != dict(physical_parameters=p.tolist(),friction=mu,actuator=act):
        raise ValueError('Use the same vehicle parameter file as the checkpoint')
    if args.mode == 'export':
        export(args,agent,metadata)
    else:
        evaluate(args,agent)


if __name__ == '__main__':
    main()
