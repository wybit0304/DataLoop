# RealCalib

- `vehicle_model.py`: complete nonlinear 8DOF body/wheel model, combined-slip tyre forces and open-loop rollout. Eight mechanical DOFs use nine first-order states, including roll angle and rate.
- `main.py`: Gaussian-process / expected-improvement optimizer, discrepancy calculation, demonstration generation and result saving. Corresponds to Methods 4.1.

From the package root:

```bash
python realcalib/main.py --demo --trials 60
python realcalib/main.py --data-dir YOUR_DATA --out YOUR_OUTPUT --trials 100
```

CSV columns: `time_s, steering_rad, torque_fl_nm, torque_fr_nm, torque_rl_nm, torque_rr_nm, vx_mps, ay_mps2, yaw_rate_radps`. Optional `vy_mps` initializes lateral velocity. Inputs are held causally; measured states are not continually injected into the rollout.

`manifest.json` supplies disjoint recording lists named `train`, `validation`, `test`. Example: `{"kind":"measured","train":["train.csv"],"validation":["val.csv"],"test":["test.csv"]}`.

The example searches 12 physical/actuator parameters with explicit bounds. It illustrates the paper's calibration procedure; it does not publish the complete final experimental search configuration. Outputs include calibrated parameters, optimization history, predicted responses, metrics and a plot. `--params` in the other modules can load the exported calibration for a new training run.
