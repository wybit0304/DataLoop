# DataLoop — core methods

Core implementation accompanying **Developing Autonomy-Native Motion Control through Data-Loop Engineering**. The three folders follow Methods 4.1–4.3. They expose the algorithms and a runnable demonstration; nominal/example parameters are supplied instead of the final experimental calibration and original vehicle datasets.

| Module | Core files | Flow |
|---|---|---|
| RealCalib | `vehicle_model.py`, `main.py` | measured inputs → 8DOF rollout → normalized error → GP/EI optimization |
| LearnVMC | `environment.py`, `td3.py`, `main.py` | Gym environment → TD3 yaw moment → QP → wheel torques |
| DeepPhy | `estimator.py`, `main.py` | sensors → 8DOF/UKF prior → causal TCN residual → estimated states |

The physical model is defined once in `realcalib/vehicle_model.py` and reused by the other modules. Keep the three folders together. Steering ratio is **21**; internal steering inputs are already road-wheel radians. Other internal quantities use SI units.

## Install and run

Tested with Python 3.11 on Windows / CPU. From this package directory:

```bash
python -m pip install -r requirements.txt
python realcalib/main.py --demo --trials 60
python learnvmc/main.py evaluate
python deepphy/main.py demo
```

Each entry point writes CSV/JSON results to its own `outputs/` directory. Small calibration/estimation datasets are generated on first use. The two `examples/` checkpoints were trained on synthetic demonstrations; they allow a quick run without retraining.

## Train, save and deploy

```bash
python learnvmc/main.py train
python learnvmc/main.py export
python learnvmc/main.py deploy
python deepphy/main.py train
```

`export` creates a reload-tested TorchScript policy taking seven raw SI observations and producing yaw moment in Nm. `deploy` combines that policy with the QP allocator and saves four-wheel torque commands from a CSV. This is the offline deployment interface. CarSim connection settings are described in the LearnVMC README.

See the short README in each folder for inputs, outputs and method details. `validation.json` records the checks performed on this version. No original real-vehicle recordings are bundled.
