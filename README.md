# DataLoop

Core methods accompanying **Developing Autonomy-Native Motion Control through Data-Loop Engineering**.

DataLoop links vehicle model calibration, motion-control policy learning and state estimation in a Real-to-Sim-to-Real workflow. This repository provides the three algorithmic modules described in Methods 4.1–4.3 of the manuscript.

## Relation to the manuscript

| Manuscript method | Directory and core files | What the code does |
|---|---|---|
| **4.1 RealCalib — simulator calibration** | `realcalib/vehicle_model.py`, `realcalib/main.py` | Rolls out a nonlinear eight-degree-of-freedom vehicle model from steering and four-wheel torque inputs. A Gaussian-process Bayesian optimizer with expected improvement identifies bounded parameters by minimizing normalized errors in longitudinal velocity, lateral acceleration and yaw rate. |
| **4.2 LearnVMC — vehicle motion control** | `learnvmc/environment.py`, `learnvmc/td3.py`, `learnvmc/main.py` | Builds a Gymnasium double-lane-change environment around the vehicle model. TD3 generates an additional yaw-moment command; a constrained quadratic-programming allocator converts the command and total torque demand into four-wheel torques. Includes training, evaluation, policy export and offline deployment. |
| **4.3 DeepPhy — state estimation** | `deepphy/estimator.py`, `deepphy/main.py` | Combines an eight-DOF/unscented Kalman filter physical estimate with a causal temporal convolutional network. The network corrects longitudinal velocity, lateral velocity and yaw rate; sideslip is reconstructed from the corrected velocities to retain a consistent four-state interface. |

The modules share `realcalib/vehicle_model.py`; keep all three directories together. RealCalib can export parameters for subsequent policy or estimator training. The estimator produces the state quantities used by LearnVMC; the demonstrations run each module separately. The LearnVMC environment also exposes an optional CarSim interface.

## Install

Use Python 3.11. The dependencies were tested on Windows with CPU execution. From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

CarSim is optional and requires a separately installed licensed solver and compatible Python interface. The built-in vehicle-model demonstrations run without CarSim.

## Run the core methods

```bash
# 1. Generate a small synthetic dataset and identify vehicle parameters
python realcalib/main.py --demo --trials 60

# 2. Train TD3, evaluate, export and replay the actor + torque allocator
python learnvmc/main.py train --steps 30000
python learnvmc/main.py evaluate
python learnvmc/main.py export
python learnvmc/main.py deploy

# 3. Train and evaluate physics-guided residual state estimation
python deepphy/main.py train --epochs 25
python deepphy/main.py demo
```

Train the networks before evaluation or export: no pretrained weights are distributed. The scripts generate small synthetic datasets where needed and save your trained checkpoints locally. Each module writes parameters, trajectories, metrics or plots to its own `outputs/` directory. LearnVMC also saves an exported TorchScript policy and a CSV of allocated wheel torques. The short README in each module describes data columns and use with your own recordings.

## Scope

This is a core-method source release with nominal/example parameters and a double-lane-change demonstration. Original vehicle recordings, final experimental calibration, pretrained weights and embedded vehicle-controller software are not included. The demonstrations illustrate the methods; they do not reproduce the manuscript's reported real-vehicle numerical results. Internal quantities use SI units; the example steering-wheel conversion uses a steering ratio of 21.
