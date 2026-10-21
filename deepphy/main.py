"""Generate a small demo, train the residual estimator, or predict on sensor CSV."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from realcalib.vehicle_model import NOMINAL, initial_state, propagate, dynamics, load_params, read_csv, save_csv, write_json, scores, plot_series
from deepphy.estimator import SENSOR_COLUMNS, REF_COLUMNS, STATE_CONFIG, ukf_branch, condition_arrays, windows, train_tcn, infer_state, append_beta, verify_checkpoint_plant

HERE = Path(__file__).resolve().parent

def make_state_demo(folder, seed=42):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    p = NOMINAL.copy()
    p[14] *= 1.12
    p[20] *= 0.9
    mu = 0.8
    for j in range(6):
        t = np.arange(0, 12.001, 0.02)
        steering = 0.025 * np.sin((0.7 + 0.08 * j) * t) * np.sin(np.pi * t / 12) ** 2
        torques = np.tile((65 + 10 * np.sin(0.6 * t))[:, None], (1, 4))
        x = initial_state(8 + 0.7 * j, p)
        rows = []
        for i, ti in enumerate(t):
            _, ax, ay, _ = dynamics(x, steering[i], torques[i], mu, p)
            imu = np.array([ax + 0.025, ay + 0.04, x[2] + 0.006]) + rng.normal(size=3) * [0.03, 0.03, 0.002]
            wheels = x[5:] * (1 + 0.008) + rng.normal(0, 0.15, 4)
            rows.append(np.r_[ti, steering[i], torques[i], imu, wheels, x[:3]])
            if i + 1 < len(t):
                x = propagate(x, steering[i], torques[i], mu, p, t[i + 1] - ti)
        save_csv(folder / f'condition_{j}.csv', np.asarray(rows), SENSOR_COLUMNS + REF_COLUMNS)
    write_json(folder / 'manifest.json', {'kind': 'synthetic', 'not_manuscript_data': True, 'generator_seed': seed, 'label_provenance': 'independent simulator truth; not real-vehicle ground truth', 'train': ['condition_0.csv', 'condition_2.csv', 'condition_4.csv'], 'validation': ['condition_1.csv'], 'test': ['condition_3.csv', 'condition_5.csv']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['demo','train','predict'],nargs='?',default='demo')
    parser.add_argument('--data-dir',type=Path,default=HERE/'data')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--out',type=Path,default=HERE/'outputs')
    parser.add_argument('--params',type=Path)
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint is None:
        args.checkpoint = args.out/'deepphy.pt'
    if args.mode != 'train' and not args.checkpoint.is_file():
        parser.error('No trained checkpoint found. Run train first with the same --out, or supply --checkpoint.')
    p,mu,_ = load_params(args.params)
    if args.mode == 'predict':
        if args.input is None:
            parser.error('predict requires --input with sensor/actuator CSV columns')
        data = read_csv(args.input,SENSOR_COLUMNS)
        window = STATE_CONFIG['window']
        if len(data) < window:
            raise ValueError(f'At least {window} causal history samples are required')
        physics = ukf_branch(data,p,mu)
        features = np.column_stack([data[k] for k in SENSOR_COLUMNS[1:]]+[physics[:,i] for i in range(3)]).astype(np.float32)
        X = np.stack([features[i-window+1:i+1] for i in range(window-1,len(data))])
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        verify_checkpoint_plant(checkpoint,p,mu)
        predicted = append_beta(infer_state(checkpoint,X,physics[window-1:].astype(np.float32)))
        save_csv(args.out/'prediction.csv',np.column_stack([data['time_s'][window-1:],predicted]),
                 ['time_s','vx_mps','vy_mps','beta_rad','yaw_rate_radps'])
        print('Saved label-free state predictions:',args.out/'prediction.csv')
        return
    manifest_path = args.data_dir/'manifest.json'
    if not manifest_path.exists():
        if args.data_dir.resolve() != (HERE/'data').resolve():
            raise FileNotFoundError('Your data directory must contain a manifest.json')
        make_state_demo(args.data_dir,args.seed)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = {k:manifest[k] for k in ['train','validation','test']}
    paths = [str((args.data_dir/n).resolve()) for seq in splits.values() for n in seq]
    if any(not v for v in splits.values()) or len(paths) != len(set(paths)):
        raise ValueError('Use disjoint, nonempty recording files for training, validation and testing')
    chunks = condition_arrays(args.data_dir,args.out/'cache',p,mu,splits)
    if args.mode == 'train':
        checkpoint = train_tcn(chunks,splits,args.out,args.epochs,args.seed)
        checkpoint.update(physical_parameters=torch.tensor(p),friction=mu,data_kind=manifest.get('kind','author_supplied'))
        torch.save(checkpoint,args.out/'deepphy.pt')
        print('Saved network and training-only normalization:',args.out/'deepphy.pt')
    else:
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    verify_checkpoint_plant(checkpoint,p,mu)
    names = ['vx_mps','vy_mps','beta_rad','yaw_rate_radps']
    report = dict(data_kind=manifest.get('kind','author_supplied'),splits=splits,test={})
    for name in splits['test']:
        X,physics,reference,t = windows(chunks[name],STATE_CONFIG['window'])
        reference = append_beta(reference)
        prediction = append_beta(infer_state(checkpoint,X,physics))
        report['test'][name] = scores(reference,prediction,names)
        save_csv(args.out/(Path(name).stem+'_prediction.csv'),np.column_stack([t,prediction]),['time_s']+names)
        plot_series(args.out/(Path(name).stem+'.png'),t,reference,prediction,names)
    write_json(args.out/'metrics.json',report)
    print(json.dumps({key:{n:v['rmse'] for n,v in values.items()} for key,values in report['test'].items()},indent=2))


if __name__ == '__main__':
    main()
