# Vehicle Path Generator from md

Generate and visualize 2D vehicle trajectories from **speed** (`v`) + **yaw-rate** (`yaw_rate`) or **steering** (`delta`) inputs.  
Includes a **unicycle model**, **kinematic bicycle model**, **noise injection**, CSV helpers, ENU↔︎LLA utilities, plotting, and a simple CLI.

> Module file: `vehicle_path.py`

---

## Features

- **Unicycle model**: `(v, yaw_rate)` with midpoint integration
- **Kinematic bicycle model**: `(v, delta, L)` where `yaw_rate = v/L * tan(delta)`
- **Noise injection** on `v`, `yaw_rate`, or `delta`
- **CSV helpers** for inputs and outputs
- **ENU ↔︎ LLA** small-area coordinate conversions
- **Matplotlib plotting** with optional heading arrows
- **CLI** for running from the command line or Spyder

---

## Installation

Place `vehicle_path.py` in your project. Optional dependencies:
```bash
pip install numpy pandas matplotlib pytest
```

In Python, add the containing folder to `sys.path` if needed:
```python
import sys
sys.path.append("path/to/folder/with/vehicle_path.py")
```

---

## Kinematics

### Continuous-time models

**Unicycle (point-mass)**
\[
\begin{aligned}
\dot{x} &= v \cos\psi,\\
\dot{y} &= v \sin\psi,\\
\dot{\psi} &= \omega,
\end{aligned}
\]
where \(v\) is forward speed [m/s] and \(\omega\) is yaw-rate [rad/s].

**Kinematic bicycle (rear-axle reference)**
\[
\begin{aligned}
\dot{x} &= v \cos\psi,\\
\dot{y} &= v \sin\psi,\\
\dot{\psi} &= \frac{v}{L}\tan\delta,
\end{aligned}
\]
where \(\delta\) is the front-wheel steering angle [rad] and \(L\) is the wheelbase [m].

> Notes: These are nonholonomic, no-slip, planar kinematics. For small \(\delta\), \(\tan\delta \approx \delta\).

### Discrete-time integration (used by the code)

We use a **midpoint (semi-implicit) update** for translation and an **explicit** update for heading,
which improves accuracy over a purely explicit Euler step at the same \(\Delta t\):

Given sample index \(k\) and step \(\Delta t_k\), let
\[
\psi_{k+\tfrac{1}{2}} = \psi_k + \tfrac{1}{2}\,\dot{\psi}_k\,\Delta t_k.
\]

- **Unicycle:**
\[
\begin{aligned}
\psi_{k+1} &= \psi_k + \omega_k\,\Delta t_k,\\
x_{k+1} &= x_k + v_k\,\Delta t_k \cos\!\big(\psi_{k+\tfrac{1}{2}}\big),\\
y_{k+1} &= y_k + v_k\,\Delta t_k \sin\!\big(\psi_{k+\tfrac{1}{2}}\big).
\end{aligned}
\]

- **Bicycle:**
\[
\begin{aligned}
\omega_k &= \frac{v_k}{L}\tan\delta_k,\\
\psi_{k+1} &= \psi_k + \omega_k\,\Delta t_k,\\
x_{k+1} &= x_k + v_k\,\Delta t_k \cos\!\big(\psi_{k+\tfrac{1}{2}}\big),\\
y_{k+1} &= y_k + v_k\,\Delta t_k \sin\!\big(\psi_{k+\tfrac{1}{2}}\big).
\end{aligned}
\]

**Timestamps vs. fixed step:**  
If timestamps \(t_k\) are provided, the code computes \(\Delta t_k = t_{k}-t_{k-1}\). Otherwise, a scalar or per-sample \(\Delta t\) can be supplied.

**Heading convention:**  
\(\psi=0\) aligns with +x, positive rotation is CCW, and positions are in a local ENU frame (meters).

**Relationship between bicycle and unicycle:**  
Feeding the unicycle with \(\omega_k = \tfrac{v_k}{L}\tan\delta_k\) yields trajectories that closely match the bicycle model (tested in the suite).

---

## Quick Start

### Unicycle model (v, yaw_rate)
```python
import vehicle_path as vp

# Build a simple example input profile
df = vp.example_s_curve(duration=15.0, dt=0.05)

# Simulate and plot
path = vp.simulate_from_dataframe_unicycle(df, init=vp.Pose(0, 0, 0))
vp.plot_path(path, pose_stride=40, title="Unicycle path")
```

### Bicycle model (v, delta, L)
```python
import vehicle_path as vp

df = vp.example_slalom(duration=18.0, dt=0.05)
path = vp.simulate_from_dataframe_bicycle(df, L=2.8, init=vp.Pose(0, 0, 0),
                                          noise_std_v=0.2, noise_std_delta=0.01)
vp.plot_path(path, pose_stride=50, title="Bicycle model (v, δ) with noise")
```

### Circular path (50m radius example)
```python
import vehicle_path as vp

# Generate a circular path with 50m radius
df = vp.example_circle(radius=50.0, v=2.0, dt=0.1, n_circles=1.0)

# Simulate and plot
path = vp.simulate_from_dataframe_unicycle(df, init=vp.Pose(0, 0, 0))
vp.plot_path(path, pose_stride=150, title="50m Radius Circle Path")
```

Or use the pre-generated CSV:
```bash
python vehicle_path.py --model unicycle --input example_circle_50m.csv --pose-stride 150
```

---

## Data Models

### Pose
```python
@dataclass
class Pose:
    x: float = 0.0    # meters
    y: float = 0.0    # meters
    yaw: float = 0.0  # radians
```

### PathResult
```python
@dataclass
class PathResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    yaw_rate: np.ndarray
```

---

## API Reference

### Unicycle
```python
simulate_unicycle(v, yaw_rate, dt=None, t=None, init=Pose(),
                  noise_std_v=0.0, noise_std_yawrate=0.0, rng=None) -> PathResult
```
- `v`, `yaw_rate`: scalar or array-like
- Provide **either** `dt` (scalar/array) **or** timestamps `t` (array)
- Noise is optional

### Bicycle
```python
simulate_bicycle(v, delta, L, dt=None, t=None, init=Pose(),
                 noise_std_v=0.0, noise_std_delta=0.0, rng=None) -> PathResult
```
- `delta`: steering angle [rad], `L`: wheelbase [m]

### From DataFrame
```python
simulate_from_dataframe_unicycle(df, init=Pose(), **kwargs) -> PathResult
simulate_from_dataframe_bicycle(df, L, init=Pose(), **kwargs) -> PathResult
```
Required columns:
- **Unicycle**: `t, v, yaw_rate`
- **Bicycle**: `t, v, delta`

### CSV Helpers
```python
read_inputs_csv(path) -> pd.DataFrame         # reads inputs with columns t,v,yaw_rate or t,v,delta
save_path_csv(path_result, out_path) -> None  # saves outputs: t,x,y,yaw,v,yaw_rate
```

### Plotting
```python
plot_path(path_result, pose_stride=0, figsize=(6,6), show=True, ax=None, title=None) -> Axes
```
- `pose_stride`: show heading arrows every N samples (0 disables).

### ENU ↔︎ LLA
```python
lat, lon = enu_to_lla(x, y, lat0_deg, lon0_deg)
x, y = lla_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg)
```
> Small-area equirectangular approximation.

### Example Generators
```python
example_s_curve(duration=15.0, dt=0.05) -> pd.DataFrame       # S-curve for unicycle
example_slalom(duration=18.0, dt=0.05) -> pd.DataFrame        # Slalom for bicycle
example_circle(radius=50.0, v=2.0, dt=0.1, n_circles=1.0) -> pd.DataFrame  # Circular path for unicycle
```

---

## CSV Formats

### Input CSV (commands)
- **Unicycle:** `t, v, yaw_rate`  
- **Bicycle:** `t, v, delta`

### Output CSV (trajectory)
```
t, x, y, yaw, v, yaw_rate
```

> Do not feed an **output** CSV back as an **input** without reconstructing `delta` or `yaw_rate` appropriately.

---

## Command Line Interface (CLI)

Run the generator directly:
```bash
python vehicle_path.py --model bicycle --wheelbase 2.7 --input inputs.csv --pose-stride 25
```

**Common options**:
- `--model {unicycle|bicycle}` (default: `unicycle`)
- `--wheelbase L` (bicycle only; default: `2.7`)
- `--input inputs.csv` (if omitted, a synthetic example is generated)
- `--duration 20.0`, `--dt 0.05` (for synthetic generation)
- `--noise-v`, `--noise-yawrate`, `--noise-delta`
- `--pose-stride 40` (heading arrows)
- `--no-plot`
- `--out path_out.csv`
- `--title "Vehicle Path"`

**Examples**:
```bash
# Unicycle with synthetic signals
python vehicle_path.py --model unicycle --duration 15 --dt 0.05 --pose-stride 40

# Bicycle from CSV
python vehicle_path.py --model bicycle --wheelbase 2.8 --input inputs.csv --pose-stride 25

# 50m radius circle path from example CSV
python vehicle_path.py --model unicycle --input example_circle_50m.csv --pose-stride 150
```

---

## Testing

A pytest suite is provided in `test_vehicle_path.py`:
- Straight-line and constant-turn unicycle
- Bicycle ≈ unicycle equivalence (`yaw_rate ≈ v/L·tan(delta)`)
- CSV round-trip and ENU↔︎LLA conversions
- Error handling for length mismatch

Run:
```bash
pytest -q test_vehicle_path.py
```

> In Spyder’s IPython console: `!pytest -q test_vehicle_path.py`

## License

MIT-style – do as you wish, attribution appreciated.

