#!/usr/bin/env python3
"""
Vehicle Path Generator from md

Generate and visualize 2D vehicle trajectories from speed (v) + yaw-rate (yaw_rate)
or steering (delta) inputs. Includes unicycle model, kinematic bicycle model,
noise injection, CSV helpers, ENU↔︎LLA utilities, plotting, and a simple CLI.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Pose:
    """Vehicle pose in 2D."""
    x: float = 0.0    # meters
    y: float = 0.0    # meters
    yaw: float = 0.0  # radians


@dataclass
class PathResult:
    """Trajectory result from simulation."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    yaw_rate: np.ndarray


# ============================================================================
# Core Simulation Functions
# ============================================================================

def simulate_unicycle(
    v: Union[float, np.ndarray],
    yaw_rate: Union[float, np.ndarray],
    dt: Optional[Union[float, np.ndarray]] = None,
    t: Optional[np.ndarray] = None,
    init: Pose = None,
    noise_std_v: float = 0.0,
    noise_std_yawrate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> PathResult:
    """
    Simulate unicycle model with midpoint integration.
    
    Parameters
    ----------
    v : float or array-like
        Forward speed [m/s]
    yaw_rate : float or array-like
        Yaw rate [rad/s]
    dt : float or array-like, optional
        Time step(s) [s]. Provide either dt or t.
    t : array-like, optional
        Timestamps [s]. Provide either dt or t.
    init : Pose, optional
        Initial pose (default: Pose())
    noise_std_v : float, optional
        Standard deviation of speed noise [m/s]
    noise_std_yawrate : float, optional
        Standard deviation of yaw rate noise [rad/s]
    rng : np.random.Generator, optional
        Random number generator for noise
    
    Returns
    -------
    PathResult
        Trajectory with t, x, y, yaw, v, yaw_rate arrays
    """
    if init is None:
        init = Pose()
    
    # Convert inputs to arrays
    v = np.atleast_1d(v)
    yaw_rate = np.atleast_1d(yaw_rate)
    
    # Determine time steps
    if t is not None:
        t = np.atleast_1d(t)
        N = len(t)
        if len(v) == 1:
            v = np.full(N, v[0])
        if len(yaw_rate) == 1:
            yaw_rate = np.full(N, yaw_rate[0])
    elif dt is not None:
        dt = np.atleast_1d(dt)
        N = max(len(v), len(yaw_rate))
        if len(v) == 1:
            v = np.full(N, v[0])
        if len(yaw_rate) == 1:
            yaw_rate = np.full(N, yaw_rate[0])
        if len(dt) == 1:
            t = np.arange(N) * dt[0]
        else:
            t = np.concatenate([[0], np.cumsum(dt)])
            if len(t) < N:
                t = np.pad(t, (0, N - len(t)), mode='edge')
            t = t[:N]
    else:
        raise ValueError("Must provide either dt or t")
    
    # Check lengths match
    if not (len(v) == len(yaw_rate) == N):
        raise ValueError(
            f"Length mismatch: v={len(v)}, yaw_rate={len(yaw_rate)}, N={N}"
        )
    
    # Add noise if requested
    if noise_std_v > 0 or noise_std_yawrate > 0:
        if rng is None:
            rng = np.random.default_rng()
        if noise_std_v > 0:
            v = v + rng.normal(0, noise_std_v, N)
        if noise_std_yawrate > 0:
            yaw_rate = yaw_rate + rng.normal(0, noise_std_yawrate, N)
    
    # Initialize output arrays
    x = np.zeros(N)
    y = np.zeros(N)
    yaw = np.zeros(N)
    
    x[0] = init.x
    y[0] = init.y
    yaw[0] = init.yaw
    
    # Midpoint integration
    for k in range(N - 1):
        dt_k = t[k + 1] - t[k]
        
        # Explicit yaw update
        yaw[k + 1] = yaw[k] + yaw_rate[k] * dt_k
        
        # Midpoint yaw for position update
        yaw_mid = yaw[k] + 0.5 * yaw_rate[k] * dt_k
        
        # Position update using midpoint yaw
        x[k + 1] = x[k] + v[k] * dt_k * np.cos(yaw_mid)
        y[k + 1] = y[k] + v[k] * dt_k * np.sin(yaw_mid)
    
    return PathResult(t=t, x=x, y=y, yaw=yaw, v=v, yaw_rate=yaw_rate)


def simulate_bicycle(
    v: Union[float, np.ndarray],
    delta: Union[float, np.ndarray],
    L: float,
    dt: Optional[Union[float, np.ndarray]] = None,
    t: Optional[np.ndarray] = None,
    init: Pose = None,
    noise_std_v: float = 0.0,
    noise_std_delta: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> PathResult:
    """
    Simulate kinematic bicycle model with midpoint integration.
    
    Parameters
    ----------
    v : float or array-like
        Forward speed [m/s]
    delta : float or array-like
        Steering angle [rad]
    L : float
        Wheelbase [m]
    dt : float or array-like, optional
        Time step(s) [s]. Provide either dt or t.
    t : array-like, optional
        Timestamps [s]. Provide either dt or t.
    init : Pose, optional
        Initial pose (default: Pose())
    noise_std_v : float, optional
        Standard deviation of speed noise [m/s]
    noise_std_delta : float, optional
        Standard deviation of steering angle noise [rad]
    rng : np.random.Generator, optional
        Random number generator for noise
    
    Returns
    -------
    PathResult
        Trajectory with t, x, y, yaw, v, yaw_rate arrays
    """
    if init is None:
        init = Pose()
    
    # Convert inputs to arrays
    v = np.atleast_1d(v)
    delta = np.atleast_1d(delta)
    
    # Determine time steps
    if t is not None:
        t = np.atleast_1d(t)
        N = len(t)
        if len(v) == 1:
            v = np.full(N, v[0])
        if len(delta) == 1:
            delta = np.full(N, delta[0])
    elif dt is not None:
        dt = np.atleast_1d(dt)
        N = max(len(v), len(delta))
        if len(v) == 1:
            v = np.full(N, v[0])
        if len(delta) == 1:
            delta = np.full(N, delta[0])
        if len(dt) == 1:
            t = np.arange(N) * dt[0]
        else:
            t = np.concatenate([[0], np.cumsum(dt)])
            if len(t) < N:
                t = np.pad(t, (0, N - len(t)), mode='edge')
            t = t[:N]
    else:
        raise ValueError("Must provide either dt or t")
    
    # Check lengths match
    if not (len(v) == len(delta) == N):
        raise ValueError(
            f"Length mismatch: v={len(v)}, delta={len(delta)}, N={N}"
        )
    
    # Add noise if requested
    if noise_std_v > 0 or noise_std_delta > 0:
        if rng is None:
            rng = np.random.default_rng()
        if noise_std_v > 0:
            v = v + rng.normal(0, noise_std_v, N)
        if noise_std_delta > 0:
            delta = delta + rng.normal(0, noise_std_delta, N)
    
    # Compute yaw rate from bicycle kinematics
    yaw_rate = (v / L) * np.tan(delta)
    
    # Initialize output arrays
    x = np.zeros(N)
    y = np.zeros(N)
    yaw = np.zeros(N)
    
    x[0] = init.x
    y[0] = init.y
    yaw[0] = init.yaw
    
    # Midpoint integration
    for k in range(N - 1):
        dt_k = t[k + 1] - t[k]
        
        # Explicit yaw update
        yaw[k + 1] = yaw[k] + yaw_rate[k] * dt_k
        
        # Midpoint yaw for position update
        yaw_mid = yaw[k] + 0.5 * yaw_rate[k] * dt_k
        
        # Position update using midpoint yaw
        x[k + 1] = x[k] + v[k] * dt_k * np.cos(yaw_mid)
        y[k + 1] = y[k] + v[k] * dt_k * np.sin(yaw_mid)
    
    return PathResult(t=t, x=x, y=y, yaw=yaw, v=v, yaw_rate=yaw_rate)


# ============================================================================
# DataFrame Wrappers
# ============================================================================

def simulate_from_dataframe_unicycle(
    df: pd.DataFrame,
    init: Pose = None,
    **kwargs
) -> PathResult:
    """
    Simulate unicycle model from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: t, v, yaw_rate
    init : Pose, optional
        Initial pose (default: Pose())
    **kwargs
        Additional arguments passed to simulate_unicycle
    
    Returns
    -------
    PathResult
        Trajectory result
    """
    if init is None:
        init = Pose()
    
    required_cols = ['t', 'v', 'yaw_rate']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    
    return simulate_unicycle(
        v=df['v'].values,
        yaw_rate=df['yaw_rate'].values,
        t=df['t'].values,
        init=init,
        **kwargs
    )


def simulate_from_dataframe_bicycle(
    df: pd.DataFrame,
    L: float,
    init: Pose = None,
    **kwargs
) -> PathResult:
    """
    Simulate bicycle model from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: t, v, delta
    L : float
        Wheelbase [m]
    init : Pose, optional
        Initial pose (default: Pose())
    **kwargs
        Additional arguments passed to simulate_bicycle
    
    Returns
    -------
    PathResult
        Trajectory result
    """
    if init is None:
        init = Pose()
    
    required_cols = ['t', 'v', 'delta']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    
    return simulate_bicycle(
        v=df['v'].values,
        delta=df['delta'].values,
        L=L,
        t=df['t'].values,
        init=init,
        **kwargs
    )


# ============================================================================
# CSV Helpers
# ============================================================================

def read_inputs_csv(path: str) -> pd.DataFrame:
    """
    Read input CSV file with vehicle commands.
    
    Parameters
    ----------
    path : str
        Path to CSV file with columns t,v,yaw_rate (unicycle) or t,v,delta (bicycle)
    
    Returns
    -------
    pd.DataFrame
        Input data
    """
    return pd.read_csv(path)


def save_path_csv(path_result: PathResult, out_path: str) -> None:
    """
    Save trajectory result to CSV.
    
    Parameters
    ----------
    path_result : PathResult
        Trajectory to save
    out_path : str
        Output CSV file path
    """
    df = pd.DataFrame({
        't': path_result.t,
        'x': path_result.x,
        'y': path_result.y,
        'yaw': path_result.yaw,
        'v': path_result.v,
        'yaw_rate': path_result.yaw_rate,
    })
    df.to_csv(out_path, index=False)


# ============================================================================
# Plotting
# ============================================================================

def plot_path(
    path_result: PathResult,
    pose_stride: int = 0,
    figsize: Tuple[int, int] = (6, 6),
    show: bool = True,
    ax = None,
    title: Optional[str] = None,
):
    """
    Plot vehicle trajectory with optional heading arrows.
    
    Parameters
    ----------
    path_result : PathResult
        Trajectory to plot
    pose_stride : int, optional
        Show heading arrows every N samples (0 disables)
    figsize : tuple, optional
        Figure size (width, height)
    show : bool, optional
        Whether to show the plot
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
    title : str, optional
        Plot title
    
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting", file=sys.stderr)
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trajectory
    ax.plot(path_result.x, path_result.y, 'b-', linewidth=1.5, label='Path')
    
    # Plot start and end points
    ax.plot(path_result.x[0], path_result.y[0], 'go', markersize=8, label='Start')
    ax.plot(path_result.x[-1], path_result.y[-1], 'ro', markersize=8, label='End')
    
    # Plot heading arrows
    if pose_stride > 0:
        arrow_scale = 0.5
        for i in range(0, len(path_result.x), pose_stride):
            dx = arrow_scale * np.cos(path_result.yaw[i])
            dy = arrow_scale * np.sin(path_result.yaw[i])
            ax.arrow(
                path_result.x[i], path_result.y[i], dx, dy,
                head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.6
            )
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title:
        ax.set_title(title)
    
    if show:
        plt.show()
    
    return ax


# ============================================================================
# ENU ↔︎ LLA Coordinate Conversions
# ============================================================================

def enu_to_lla(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert ENU (local) to LLA (lat/lon) using equirectangular approximation.
    
    Parameters
    ----------
    x : float or array-like
        East coordinate [m]
    y : float or array-like
        North coordinate [m]
    lat0_deg : float
        Reference latitude [degrees]
    lon0_deg : float
        Reference longitude [degrees]
    
    Returns
    -------
    lat_deg : float or array-like
        Latitude [degrees]
    lon_deg : float or array-like
        Longitude [degrees]
    """
    # Earth radius
    R = 6378137.0  # meters
    
    # Convert reference to radians
    lat0_rad = np.deg2rad(lat0_deg)
    
    # Convert ENU to lat/lon
    dlat = y / R
    dlon = x / (R * np.cos(lat0_rad))
    
    lat_deg = lat0_deg + np.rad2deg(dlat)
    lon_deg = lon0_deg + np.rad2deg(dlon)
    
    return lat_deg, lon_deg


def lla_to_enu(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert LLA (lat/lon) to ENU (local) using equirectangular approximation.
    
    Parameters
    ----------
    lat_deg : float or array-like
        Latitude [degrees]
    lon_deg : float or array-like
        Longitude [degrees]
    lat0_deg : float
        Reference latitude [degrees]
    lon0_deg : float
        Reference longitude [degrees]
    
    Returns
    -------
    x : float or array-like
        East coordinate [m]
    y : float or array-like
        North coordinate [m]
    """
    # Earth radius
    R = 6378137.0  # meters
    
    # Convert to radians
    lat0_rad = np.deg2rad(lat0_deg)
    dlat = np.deg2rad(lat_deg - lat0_deg)
    dlon = np.deg2rad(lon_deg - lon0_deg)
    
    # Convert to ENU
    y = dlat * R
    x = dlon * R * np.cos(lat0_rad)
    
    return x, y


# ============================================================================
# Example Data Generators
# ============================================================================

def example_s_curve(duration: float = 15.0, dt: float = 0.05) -> pd.DataFrame:
    """
    Generate example S-curve trajectory for unicycle model.
    
    Parameters
    ----------
    duration : float, optional
        Duration of trajectory [s]
    dt : float, optional
        Time step [s]
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t, v, yaw_rate
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Forward speed with slight variation
    v = 2.0 + 0.5 * np.sin(2 * np.pi * t / duration)
    
    # Yaw rate creating S-curve
    yaw_rate = 0.3 * np.sin(2 * np.pi * t / (duration / 2))
    
    return pd.DataFrame({'t': t, 'v': v, 'yaw_rate': yaw_rate})


def example_slalom(duration: float = 18.0, dt: float = 0.05) -> pd.DataFrame:
    """
    Generate example slalom trajectory for bicycle model.
    
    Parameters
    ----------
    duration : float, optional
        Duration of trajectory [s]
    dt : float, optional
        Time step [s]
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t, v, delta
    """
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Constant forward speed
    v = np.full(N, 3.0)
    
    # Steering angle creating slalom pattern
    delta = 0.15 * np.sin(2 * np.pi * t / 3.0)
    
    return pd.DataFrame({'t': t, 'v': v, 'delta': delta})


def example_circle(radius: float = 50.0, v: float = 2.0, 
                   dt: float = 0.1, n_circles: float = 1.0) -> pd.DataFrame:
    """
    Generate circular path trajectory for unicycle model.
    
    Parameters
    ----------
    radius : float, optional
        Circle radius [m] (default: 50.0)
    v : float, optional
        Forward speed [m/s] (default: 2.0)
    dt : float, optional
        Time step [s] (default: 0.1)
    n_circles : float, optional
        Number of circles to complete (default: 1.0)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: t, v, yaw_rate
    """
    # For circular motion: yaw_rate = v / radius
    yaw_rate = v / radius
    
    # Calculate duration to complete n_circles
    # Full circle is 2*pi radians
    duration = n_circles * 2 * np.pi / yaw_rate
    
    # Generate time array
    t = np.arange(0, duration, dt)
    N = len(t)
    
    # Constant speed and yaw rate
    v_arr = np.full(N, v)
    yaw_rate_arr = np.full(N, yaw_rate)
    
    return pd.DataFrame({'t': t, 'v': v_arr, 'yaw_rate': yaw_rate_arr})


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Command line interface for vehicle path generator."""
    parser = argparse.ArgumentParser(
        description='Generate and visualize 2D vehicle trajectories'
    )
    
    parser.add_argument(
        '--model',
        choices=['unicycle', 'bicycle'],
        default='unicycle',
        help='Vehicle model to use'
    )
    parser.add_argument(
        '--wheelbase', '-L',
        type=float,
        default=2.7,
        help='Wheelbase for bicycle model [m]'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=20.0,
        help='Duration for synthetic example [s]'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.05,
        help='Time step for synthetic example [s]'
    )
    parser.add_argument(
        '--noise-v',
        type=float,
        default=0.0,
        help='Standard deviation of speed noise [m/s]'
    )
    parser.add_argument(
        '--noise-yawrate',
        type=float,
        default=0.0,
        help='Standard deviation of yaw rate noise [rad/s]'
    )
    parser.add_argument(
        '--noise-delta',
        type=float,
        default=0.0,
        help='Standard deviation of steering angle noise [rad]'
    )
    parser.add_argument(
        '--pose-stride',
        type=int,
        default=40,
        help='Show heading arrows every N samples (0 disables)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--out',
        type=str,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Plot title'
    )
    
    args = parser.parse_args()
    
    # Load or generate input data
    if args.input:
        df = read_inputs_csv(args.input)
        print(f"Loaded input from {args.input}")
    else:
        if args.model == 'unicycle':
            df = example_s_curve(duration=args.duration, dt=args.dt)
            print(f"Generated synthetic unicycle example (duration={args.duration}s, dt={args.dt}s)")
        else:
            df = example_slalom(duration=args.duration, dt=args.dt)
            print(f"Generated synthetic bicycle example (duration={args.duration}s, dt={args.dt}s)")
    
    # Simulate
    init = Pose(0, 0, 0)
    
    if args.model == 'unicycle':
        path = simulate_from_dataframe_unicycle(
            df,
            init=init,
            noise_std_v=args.noise_v,
            noise_std_yawrate=args.noise_yawrate,
        )
        model_name = "Unicycle"
    else:
        path = simulate_from_dataframe_bicycle(
            df,
            L=args.wheelbase,
            init=init,
            noise_std_v=args.noise_v,
            noise_std_delta=args.noise_delta,
        )
        model_name = f"Bicycle (L={args.wheelbase}m)"
    
    print(f"Simulated {len(path.t)} steps")
    
    # Save output if requested
    if args.out:
        save_path_csv(path, args.out)
        print(f"Saved output to {args.out}")
    
    # Plot
    if not args.no_plot:
        title = args.title or f"{model_name} Path"
        plot_path(path, pose_stride=args.pose_stride, title=title)


if __name__ == '__main__':
    main()
