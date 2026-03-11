# Spot-RL
Code & Dockerfile for Spot Reinforcement Learning demo

# Import our image from .tar
```bash
docker load -i spot-rl-demo-<arch>.tar
docker tag spot-rl-demo:<arch> spot-rl-demo:latest
```

# Default Model 15k Steps
```bash
docker run --privileged --rm -it -v /dev/input:/dev/input spot-rl-demo:latest <ip of robot api> /spot-rl/external/models/
````

# Bring your own model (don't forget to set the IP)
```bash
docker run --privileged --rm -it -v /dev/input:/dev/input -v /path/to/folder/with/onz:/models spot-rl-demo:latest 192.168.x.y /models
```

# Example with local directory ./Model_Under_Test (don't forget to set the IP)
```bash
docker run --privileged --rm -it -v /dev/input:/dev/input -v ./Model_Under_Test/:/mut spot-rl-demo:latest 192.168.x.y /mut
```

# Installing without docker from locally cloned repo
```bash
sudo apt update
sudo apt install python3-pip
pip3 install gitman
gitman update
cd external/spot_python_sdk/prebuilt
pip3 install bosdyn_api-4.0.0-py3-none-any.whl
pip3 install bosdyn_core-4.0.0-py3-none-any.whl
pip3 install bosdyn_client-4.0.0-py3-none-any.whl
pip3 install pygame
pip3 install pyPS4Controller
pip3 install spatialmath-python
pip3 install onnxruntime
```

# Example of mocked
```bash
uv run rl_deploy/spot_rl_demo.py  0000 --mock
```

# Plot Acquisition Frequencies
```bash
uv run rl_deploy/plot_acquisition_frequencies.py --hdf5_files spot_isaac_real.hdf5
```

# Timing Diagnostics (dt_*)
The codebase tracks several `dt` variables to monitor system timings and ensure it meets real-time requirements. For a 50Hz system, the expected cycle time is 0.02s (20ms).
- `dt_acquisition_timestamp`: Time between consecutive state observations from the robot. Evaluated from `RobotStateStreamResponse`. Expected: **0.02s**.
- `dt_total_step`: Total latency of the entire control step cycle. It represents the time between consecutive calls to the `OnnxCommandGenerator`. Expected: **~0.02s**.
- `dt_divider_wait`: Time spent waiting for the `EventDivider` to hit the required trigger factor (waiting for new robot states). Expected: **Varies, typically closely follows the period of the state arrivals ~0.02s**.
- `dt_divider_to_onnx`: The gap from when the `EventDivider` finishes waiting until the `OnnxCommandGenerator` starts computing the action. Expected: **Very small (< 0.001s)**.
- `dt_state_arrival_to_compute`: Latency from the exact moment the state arrives via callback in the `StateHandler` until the ONNX action computation starts. Expected: **Very small (< 0.002s)**.
- `dt_onnx_compute`: Time spent strictly running the ONNX model inference. Expected: **Depends on the model, usually < 0.005s**.
- `dt_post_process`: Time taken to process the raw output of the ONNX model into a Spot-ready command. Expected: **Very small (< 0.001s)**.
- `dt_command_latency`: Total round-trip latency from when the robot captures state/receives request to when it receives the corresponding command (computed as `last_command.received_timestamp[i] - header.request_received_timestamp[i-1]`). Expected: **~0.02s - 0.04s**.

To check these statistics across a pre-logged dataset (excluding the turbulent first and last seconds), run:
```bash
uv run rl_deploy/print_timing_diagnostics.py --hdf5_files spot_isaac_real.hdf5
```

Example output:
```
--- Timing Diagnostics for spot_isaac_real.hdf5 ---
Warning: Dataset is too short to skip 1s at start and end. Using all data.
Total samples: 1131, Valid samples used for average: 1131
Variable Name                  | Average (s)     | Expected (s)   
-----------------------------------------------------------------
dt_acquisition_timestamp       | 0.018008        | approx 0.02    
dt_total_step                  | 0.017996        | approx 0.02    
dt_divider_wait                | 0.016983        | < 0.02 (varies)
dt_divider_to_onnx             | 0.000005        | < 0.001        
dt_state_arrival_to_compute    | 0.001178        | < 0.002        
dt_onnx_compute                | 0.000045        | < 0.005        
dt_post_process                | 0.000008        | < 0.001        
```
