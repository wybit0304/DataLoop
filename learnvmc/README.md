# LearnVMC / TD3

- `environment.py`: standard `gymnasium.Env`, DLC reference, complete 8DOF plant interface, bicycle references, reward, bounded QP torque allocation and optional `CarSimBackend`.
- `td3.py`: actor, twin critics, replay, target-policy smoothing, delayed updates, checkpoint saving/loading and physical-unit TorchScript export.
- `main.py`: training, validation selection, evaluation, export and offline deployment. Corresponds to Methods 4.2.

```bash
python learnvmc/main.py train --steps 30000 --out learnvmc/my_run
python learnvmc/main.py evaluate --out learnvmc/my_run
python learnvmc/main.py export --out learnvmc/my_run
python learnvmc/main.py deploy --out learnvmc/my_run
```

`reset(seed=..., options=...)` returns `(observation, info)`; `step(action)` returns `(observation, reward, terminated, truncated, info)`. `options['scenario']` accepts speed, friction, mass factor and DLC geometry. Observation order: steering, vx, heading error, beta, yaw rate, beta error, yaw-rate error. The environment normalizes these seven values; scalar action ±1 maps to ±3500 Nm. QP maps requested total torque/yaw moment to four feasible wheel torques.

Training saves the selected actor, critics, targets and optimizers to `td3.pt`, plus `training.csv` and `validation.csv`. The replay buffer is not stored. Evaluation saves trajectories, metrics and `deployment_inputs.csv`. Export saves `policy.ts` and its SI input/output description `policy.json`; deployment reads those files and the CSV, then writes `deployed_torques.csv`. To deploy your own states, pass `--input YOUR_CSV`; its columns match `deployment_inputs.csv`.

No pretrained weights are included. Train first; the selected output directory's `td3.pt` is then used automatically. `--checkpoint` selects your own compatible checkpoint. These commands implement the policy-learning and deployment flow in Methods 4.2 with example parameters and a DLC scenario.

## CarSim bridge

Copy and edit `carsim_config.example.json`: set the CarSim database and local `pycarsimlib` directories, channel order and conversion scales. The adapter uses the `CarsimManager` interface from the supplied source. Configure **5 IMPORT channels** (FL/FR/RL/RR torque and steering) and **12 EXPORT channels**, in exactly the JSON order. Match the project solver step to the 0.05 s controller period. The steering example converts road-wheel radians to steering-wheel degrees using ratio 21. Check all export units against your project.

```bash
python learnvmc/main.py evaluate --out learnvmc/my_run --carsim-config YOUR_CONFIG.json
```

The bridge recreates the solver on reset, validates channels/time steps and closes the session. The CarSim project defines its actual vehicle and initial state; built-in mass randomization is disabled for this backend. Interface tests use a mock manager; a licensed live CarSim run has not been verified here. The default 8DOF backend runs without CarSim.

CarSim is optional: its licensed solver and the supplied `pycarsimlib` package/dependencies must be installed separately. Pass `--params` when the external vehicle's geometry differs from the nominal example.
