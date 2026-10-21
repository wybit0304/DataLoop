# DeepPhy state estimation

- `estimator.py`: 8DOF UKF physical prior, causal dilated TCN, three-output residual loss, preprocessing normalization and inference. Corresponds to Methods 4.3.
- `main.py`: generate a small demo, train/save the estimator or predict from a sensor CSV. Reuses `realcalib/vehicle_model.py`.

```bash
python deepphy/main.py train --epochs 25
python deepphy/main.py demo
python deepphy/main.py predict --input YOUR_CSV --checkpoint deepphy/outputs/deepphy.pt
```

Sensor columns: `time_s, steering_rad, torque_fl_nm, torque_fr_nm, torque_rl_nm, torque_rr_nm, ax_mps2, ay_mps2, yaw_rate_sensor_radps, omega_fl_radps, omega_fr_radps, omega_rl_radps, omega_rr_radps`. Training additionally uses `vx_ref_mps, vy_ref_mps, yaw_rate_ref_radps`. Wheel speed is wheel rad/s; convert motor units before importing. Reference labels are excluded from prediction inputs.

The UKF processes current/past IMU and wheel measurements. The TCN learns residuals for **vx, vy and yaw rate** from a 50-sample causal window; beta is reconstructed as `atan2(vy,max(vx,0.5))`. The loss combines normalized state error, beta consistency and residual regularization.

Use a `manifest.json` with whole-recording `train/validation/test` lists. Scalers are fitted on training data; validation selects weights. The saved checkpoint includes normalization and physical parameters. Outputs are a checkpoint, metrics, plots and four-state CSV predictions; the first 49 samples are history warm-up. No pretrained weights are included: run `train` before `demo` or `predict`. The built-in data generator illustrates the method using synthetic signals and example parameters.
