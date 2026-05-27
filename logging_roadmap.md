# What to record from the real robot

For the Spot deployment, record:

**Per experiment run:**
- Rosbag with: joint positions, joint velocities, joint torques, IMU data, base pose (from state estimator), commanded velocities, foot contact states
- Video from external camera (for qualitative gait analysis)
- Inference latency per control step (measure on the onboard compute)

**Analysis we'll do post-hoc on the rosbag:**
- Gait diagrams (foot contact timing patterns)
- Tracking error (commanded vs. actual velocity)
- Base orientation stability (roll/pitch variance)
- Energy efficiency (sum of |torque × velocity| over time)
- Recovery behavior after manual pushes (if we include perturbation tests)

we don't need to do any representation analysis on the real robot — that's all from simulation. The real robot validates whether the performance and behavior differences observed in simulation survive transfer.

**Metrics extracted from rosbag (post-hoc):**

- **Gait diagrams:** Foot contact on/off timing for all 4 legs, visualized as horizontal bars. Expect deeper/lower-rank policies to produce more symmetric, periodic patterns.
- **Tracking error:** RMSE between commanded and estimated base velocity over the 30-second window.
- **Stability:** Standard deviation of roll and pitch angles during steady-state locomotion.
- **Energy proxy:** Integral of |τ · q̇| (joint torque × joint velocity) over time.
- **Payload recovery:** Time from perturbation onset to base orientation returning within 2° of pre-push value.
- **Inference latency:** Assessing whether deeper networks are deployable at the required control frequency (~50 Hz for Spot).
- **High-frequency ratio as a gait quality metric**. LPN [https://arxiv.org/pdf/2602.18312] defines a High Frequency Ratio (energy above 10 Hz / total energy in joint angle outputs) as a policy smoothness metric. This is a gait quality signal you're not currently tracking in your Q3 Spot experiments, and it's directly extractable from rosbag joint data. A smoother policy (lower HFR) is consistent with lower Jacobian rank — if your low-rank SimBa architectures also produce lower HFR, it empirically links rank to gait quality in a way reviewers can immediately understand


Compressed IDEAS comments
tex
% REGIME (Moalla et al.): Legged locomotion = dense reward + high-DoF continuous control.
% Dense rewards mean advantage estimates shift continuously -> persistent trust region pressure.
% Representation collapse destroys feature geometry that anchored old advantages; clipping
% operates in probability-ratio space, NOT representation space -> collapse causes large
% semantic policy shifts even when rho_t stays in [1-eps, 1+eps]. This is the core failure mode.

% RANK HIERARCHY: r_Jac <= erank(F_theta) <= min_l erank(W_l). Weight rank hard-caps
% achievable feature rank; feature rank can be high even if Jacobian rank is collapsed.
% Gap Delta = n_DoF - r_Jac quantifies latent action directions never explored -> predictor
% of gait diversity ceiling. Optimal target: r_Jac ~ n_DoF (each joint = independent I/O direction).

% PHASE CONFOUND: cyclic gait makes erank(J_bar) artificially low without phase conditioning.
% Phase-conditioned erank(J_{theta,p}) is a necessity for this task class, not just an option.
% Standard rank tools mislead here; phase-aware analysis is the fix.
Narrative framing: "cyclic gait breaks usual tools"
I really like this angle — it gives the paper a sharper identity beyond "we scaled deeper networks." The story becomes:
Standard representation diagnostics were designed for stationary or near-stationary settings. Legged locomotion's cyclic gait structure induces periodic non-stationarity that makes naïve rank measurements misleading — a collapsed-looking metric may just be phase aliasing. We show that phase-conditioned analysis recovers the true signal, and that this signal predicts training health and sim-to-real transfer.
This reframes your contribution: not just "rank matters for locomotion," but "here's why existing rank tools fail for locomotion and here's the fix." That's a much stronger hook for the robotics community.
Data to collect
From real-world deployment:
Jacobian rank on hardware rollout vs. sim — does the rank survive sim-to-real?
Per-phase rank on hardware (use IMU/contact sensors to label stance/swing)
Compare Δ\DeltaΔ on hardware vs. sim — if it widens, that's a sim-to-real representation gap diagnostic
Joint activation variance per phase on hardware — proxy for DoF utilization without needing gradients
Legged Locomotion section — bullet outline in comments
tex
% SECTION STRUCTURE — Legged Locomotion
%
% [1] TASK DEFINITION
%   - MDP instantiation: obs = proprioceptive state (joint pos, vel, IMU, commands),
%     action = joint position targets, reward = dense locomotion reward (velocity tracking
%     + style terms + survival), n_DoF joints (state dimension, action dimension)
%   - Regime: dense reward, continuous high-DoF control, cyclic gait structure
%   - Why this regime is hard for PPO: dense rewards = continuous non-stationarity,
%     trust region pressure never fully relaxes, unlike sparse-reward episodic tasks
%
% [2] CYCLIC GAIT AS A DIAGNOSTIC CONFOUND
%   - Locomotion observations are phase-structured: stance, swing, (double support)
%   - Naïve erank(J_bar) averages over phases -> artificially low rank estimate
%     (destructive interference of phase-specific sensitivity patterns)
%   - This is specific to cyclic locomotion; not a problem in manipulation or locomotion-agnostic RL
%   - Solution: phase-conditioned Jacobian rank erank(J_{theta,p}) using contact/IMU labels
%   - Claim: standard tools mislead; phase-aware analysis is necessary for this task class
%
% [3] RANK TARGETS FOR LOCOMOTION
%   - Optimal Jacobian rank target: r_Jac ~ n_DoF (each joint should be an independent I/O direction)
%   - Gap Delta = n_DoF - r_Jac: mechanistic measure of joint underutilization
%   - Rank hierarchy: r_Jac <= erank(F_theta) <= min_l erank(W_l)
%   - This gives a principled target instead of just relative comparison across architectures
%
% [4] SIM-TO-REAL SETUP
%   - Hardware: Spot, blind locomotion, flat + uneven terrain
%   - Sim: IsaacLab, domain randomization, curriculum
%   - What we track at deployment: r_Jac on hardware rollout, phase-conditioned rank,
%     Delta vs. sim baseline
%   - Rank as a sim-to-real diagnostic: widening Delta signals representation gap
The cyclic gait framing is your sharpest angle — I'd make it the organizing principle of this section and let it motivate why phase-conditioned Jacobian rank is the key methodological contribution on the analysis side.

---

## Sugestion 

obs: the following references must be out of date or misleading, they are just for reference

Here is the updated logging plan with **Boston Dynamics Spot low-level API references** woven in at every relevant point.

***

## Spot Low-Level Control & API References

Before the protocol, here are the exact documents you will need:

| Reference | What It Gives You | Link |
|-----------|-------------------|------|
| **Joint Control API (Beta)** | High-rate, low-latency RPC streams for direct joint command; requires a Joint Control license | [dev.bostondynamics.com/docs/concepts/joint_control/README.html](https://dev.bostondynamics.com/docs/concepts/joint_control/README.html)  [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/README.html) |
| **Joint Control Examples (Python)** | `noarm_squat.py`, `wiggle_arm.py` — reference patterns for the low-level RPC stream | [dev.bostondynamics.com/python/examples/joint_control/README.html](https://dev.bostondynamics.com/python/examples/joint_control/README.html)  [dev.bostondynamics](https://dev.bostondynamics.com/python/examples/joint_control/README.html) |
| **Supplemental Data** | Gear ratios, max motor torques, knee linkage variable transmission, coupled SH1/EL0 Jacobian | [dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html)  [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html) |
| **Spot RL Researcher Kit + Paper (Miller et al.)** | First public end-to-end RL deployment on Spot using the low-level API; covers the exact deployment pipeline and SDK setup | [rai-inst.com/resources/papers/high-performance-reinforcement-learning-on-spot/](https://rai-inst.com/resources/papers/high-performance-reinforcement-learning-on-spot/)  [rai-inst](https://rai-inst.com/resources/papers/high-performance-reinforcement-learning-on-spot/), [arXiv:2504.17857](https://arxiv.org/html/2504.17857v1)  [arxiv](https://arxiv.org/html/2504.17857v1) |
| **robot_state.proto** | Motor temperatures, joint state names, degree-of-freedom naming conventions | [GitHub: spot-sdk/protos/bosdyn/api/robot_state.proto](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)  [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto) |

***

## Detailed Logging Protocol (Real Robot)

### 1. Control Interface & Timing

You will stream commands through the **Joint Control API**. Log: [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/README.html)

- **API mode** — `JointControl` stream (not high-level `RobotCommand` velocity mode)
- **Stream rate** — your target Hz and the **actual achieved loop time** from `time.time()` on the Jetson/NUC side
- **RPC latency** — round-trip time from command issue to actuator acknowledge (the API exposes this)
- **License check** — verify the robot has the Joint Control license enabled before each run [dev.bostondynamics](https://dev.bostondynamics.com/python/examples/joint_control/README.html)

**Citation for your methods section:** *"Policies are deployed via the Boston Dynamics Joint Control API, a high-rate, low-latency RPC stream for direct joint-level control (requires controlled-access license) [SDK docs]. We follow the deployment pipeline of Miller et al. [arXiv:2504.17857], the first public end-to-end RL deployment on Spot using this low-level interface."*

### 2. Joint State (Log at Stream Rate: 200–500 Hz)

Pull from the **Joint Control API feedback stream** and the **RobotStateService**: [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)

- **Joint positions** \(q_t\) and **velocities** \(\dot{q}_t\) — names must match the proto exactly: `fl.hx`, `fl.hy`, `fl.kn`, `fr.hx`, `fr.hy`, `fr.kn`, `hl.hx`, `hl.hy`, `hl.kn`, `hr.hx`, `hr.hy`, `hr.kn` (hip X, hip Y, knee)
- **Motor torques** — actual applied torque, not just command; note the knee uses a ball-screw/push-rod linkage with **variable transmission ratio** [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html)
- **Motor temperatures** — from `robot_state.proto` `MotorState` message; critical for long runs on foam where legs work harder [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)

**Important:** Spot's knee joint (`kn`) has a variable max torque due to linkage geometry. It is strongest at mid-range and weakest at full flex/extend. Log the **knee angle** alongside torque so you can normalize by the Supplemental Data transmission ratio table if you want to compare motor effort across leg configurations. [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html)

### 3. Action / Command Output

- **Position target** or **torque command** sent to each joint via the `JointControl` stream
- **Network forward-pass latency** — from observation vector ready to command vector dispatched
- **Command clipping/saturation flags** — did any joint hit the `Max motor torque` limits from the Supplemental Data table? [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html)

Gear ratios to know for your write-up: [dev.bostondynamics](https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html)
- HX / HY: 51:1, max motor torque 0.88 Nm
- KN: variable transmission, max motor torque 1.50 Nm
- SH0: 101:1, 0.89 Nm (if using arm, but you probably aren't)

### 4. Contact & Phase Detection (Critical for Your Paper)

Spot's **feet do not have explicit binary contact sensors** in the basic proto. You must derive contact from:

- **Foot force estimate** from leg Jacobian + motor torque (industry standard on Spot)
- Or **shock data** from the IMU packet if available

Log:
- **Per-foot contact force** \(F_z\) and threshold used for binary classification
- **Contact state** (`stance` / `swing`) per foot
- **Phase label** per leg: `stance`, `swing`, `double_support`, `flight` (if you achieve aerial phase like Miller et al. ) [rai-inst](https://rai-inst.com/resources/papers/high-performance-reinforcement-learning-on-spot/)
- **Gait cycle period** \(\tau_{\text{cycle}}\) and normalized phase \(\phi \in [0,1]\)

**Methodological note:** You need this for the phase-conditioned Jacobian rank \(\operatorname{erank}(\bar{J}_{\theta,p})\) in your paper. If you only log naïve \(\bar{J}_\theta\), the cyclic structure will alias and artificially lower the rank — this is your central claim.

### 5. Body State & Velocity Tracking

From the **RobotStateService** and your own estimator:
- **Body twist** \(v_{\text{body}} = (v_x, v_y, \omega_z)\) — from Spot's EKF or your own Lidar/IMU integration
- **Commanded velocity** \(v_{\text{cmd}}(t)\) — the exact reference you sent that timestep
- **Tracking error** \(e_v(t) = v_{\text{cmd}} - v_{\text{body}}\)
- **Body roll, pitch, yaw** — orientation from `RobotState.kinematic_state.body`
- **IMU data** — angular velocity, linear acceleration (from `robot_state.proto` `InertialState` ) [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)

### 6. Terrain-Specific Logs (Flat vs. Foam)

**Foam ground only:**
- **Sink depth estimate** — difference between expected touchdown height (from leg kinematics) and actual body height
- **Vertical oscillation energy** — integrate body \(a_z\) after touchdown; foam is viscoelastic
- **Stance time asymmetry** — left vs. right stance duration; foam induces timing skew you can quantify
- **Slip detection** — horizontal foot displacement during supposed stance (derived from joint angles + body pose)

**Log these same quantities on flat** so the comparison is direct and normalized.

### 7. Network Diagnostics for Rank Analysis

You cannot backprop on the Jetson easily during deployment. Options:

- **Log full observation vectors** \(x_t\) and **policy outputs** \(\mu_t, \sigma_t\) at 50–100 Hz
- **Log penultimate activations** \(F_\theta(x_t)\) from the network forward pass
- **Post-hoc Jacobian:** Run your policy `.pt` file offline on the logged \(x_t\) sequence to compute \(J_\theta(x_i)\) in PyTorch. This is the practical way to get hardware Jacobian ranks without hardware autograd.

If you have compute headroom on the Jetson, you could also log:
- **Per-step Jacobian rank** \(\operatorname{erank}(J_\theta(x_t))\) computed with `torch.autograd.functional.jacobian` on a CPU buffer — expensive, but useful for real-time monitoring.

### 8. Safety & Run Metadata

- **E-stop triggers** — timestamp, cause, phase at trigger
- **Fall / recovery events** — timestamp, body orientation preceding fall
- **Motor temperature alerts** — from `robot_state.proto`; foam runs will run hotter [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)
- **Run termination cause** — `completed`, `fall`, `estop`, `thermal_limit`, `drift_exceeded`

***

## Suggested Minimal ROS Topic / Protobuf Mapping

If you are using the Spot SDK Python library, subscribe to:

- `robot_state` → body pose, IMU, joint states, motor temps [github](https://github.com/boston-dynamics/spot-sdk/blob/master/protos/bosdyn/api/robot_state.proto)
- `joint_control` stream → your own outgoing command vector (log what you send)
- `foot_force_estimate` → derived from `JointState` torque + Jacobian, or use the `behavior_fault` / `foot_state` if available

For paper reproducibility, note the **SDK version** (e.g., `4.0.2+` for Joint Control API ) and whether you use the **Spot RL Researcher Kit** pipeline from Miller et al.. [dev.bostondynamics](https://dev.bostondynamics.com/python/examples/joint_control/README.html)

***

## What to Collect per Experimental Condition

| Condition | Number of Runs | Duration Each |
|-----------|---------------|---------------|
| Flat ground, slow speed | ≥ 10 | 60 s |
| Flat ground, max speed | ≥ 10 | 60 s |
| Flat ground, command steps | ≥ 5 | 30 s |
| Foam ground, slow speed | ≥ 10 | 60 s |
| Foam ground, max speed | ≥ 10 | 60 s |
| Foam ground, command steps | ≥ 5 | 30 s |

**Policy comparisons:** shallow net vs. deep net, same conditions, same metrics.

Do you want me to now draft the actual Python logger skeleton using the Spot SDK `JointControl` stream and the proto definitions?