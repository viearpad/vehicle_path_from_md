#!/usr/bin/env python3
"""
Test suite for vehicle_path.py

Tests:
- Straight-line and constant-turn unicycle
- Bicycle ≈ unicycle equivalence (yaw_rate ≈ v/L·tan(delta))
- CSV round-trip and ENU↔︎LLA conversions
- Error handling for length mismatch
"""

import tempfile
import os
import numpy as np
import pandas as pd
import pytest

import vehicle_path as vp


class TestDataModels:
    """Test data model classes."""
    
    def test_pose_default(self):
        """Test default Pose initialization."""
        pose = vp.Pose()
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.yaw == 0.0
    
    def test_pose_custom(self):
        """Test custom Pose initialization."""
        pose = vp.Pose(x=1.0, y=2.0, yaw=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.yaw == 0.5
    
    def test_path_result(self):
        """Test PathResult creation."""
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0])
        yaw = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 1.0, 1.0])
        yaw_rate = np.array([0.0, 0.0, 0.0])
        
        result = vp.PathResult(t=t, x=x, y=y, yaw=yaw, v=v, yaw_rate=yaw_rate)
        
        assert len(result.t) == 3
        assert len(result.x) == 3
        np.testing.assert_array_equal(result.t, t)


class TestUnicycleStraightLine:
    """Test unicycle model with straight-line motion."""
    
    def test_straight_line_constant_speed(self):
        """Test straight-line motion at constant speed."""
        v = 2.0  # m/s
        yaw_rate = 0.0  # rad/s
        dt = 0.1  # s
        N = 10
        
        result = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Check that we moved in a straight line along x-axis
        expected_x = np.arange(N) * v * dt
        np.testing.assert_allclose(result.x, expected_x, rtol=1e-10)
        np.testing.assert_allclose(result.y, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.yaw, 0.0, atol=1e-10)
    
    def test_straight_line_angled(self):
        """Test straight-line motion at an angle."""
        v = 1.0  # m/s
        yaw_rate = 0.0  # rad/s
        dt = 1.0  # s
        init_yaw = np.pi / 4  # 45 degrees
        N = 5
        
        result = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, init_yaw)
        )
        
        # Should move at 45 degree angle
        expected_x = np.arange(N) * v * dt * np.cos(init_yaw)
        expected_y = np.arange(N) * v * dt * np.sin(init_yaw)
        
        np.testing.assert_allclose(result.x, expected_x, rtol=1e-6)
        np.testing.assert_allclose(result.y, expected_y, rtol=1e-6)
        np.testing.assert_allclose(result.yaw, init_yaw, atol=1e-10)


class TestUnicycleConstantTurn:
    """Test unicycle model with constant turning."""
    
    def test_constant_turn_circular(self):
        """Test circular motion with constant turn rate."""
        v = 1.0  # m/s
        yaw_rate = 0.5  # rad/s
        dt = 0.1  # s
        N = 100
        
        result = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Check that yaw increases linearly
        expected_yaw = np.arange(N) * yaw_rate * dt
        np.testing.assert_allclose(result.yaw, expected_yaw, rtol=1e-6)
        
        # For circular motion, radius should be v/yaw_rate
        radius = v / yaw_rate
        
        # After full circle, should return near starting position
        # (with some numerical error due to discrete integration)
        angle_traveled = yaw_rate * dt * (N - 1)
        if angle_traveled >= 2 * np.pi:
            # Note: won't be exact due to discrete integration
            assert result.x[-1] < radius  # stayed within reasonable bounds
            assert result.y[-1] < radius


class TestBicycleModel:
    """Test bicycle model."""
    
    def test_bicycle_zero_steering(self):
        """Test bicycle with zero steering (straight line)."""
        v = 2.0  # m/s
        delta = 0.0  # rad
        L = 2.7  # m
        dt = 0.1  # s
        N = 10
        
        result = vp.simulate_bicycle(
            v=np.full(N, v),
            delta=np.full(N, delta),
            L=L,
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Should move in straight line
        expected_x = np.arange(N) * v * dt
        np.testing.assert_allclose(result.x, expected_x, rtol=1e-10)
        np.testing.assert_allclose(result.y, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.yaw, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.yaw_rate, 0.0, atol=1e-10)
    
    def test_bicycle_constant_steering(self):
        """Test bicycle with constant steering."""
        v = 2.0  # m/s
        delta = 0.1  # rad
        L = 2.7  # m
        dt = 0.05  # s
        N = 50
        
        result = vp.simulate_bicycle(
            v=np.full(N, v),
            delta=np.full(N, delta),
            L=L,
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Yaw rate should be constant
        expected_yaw_rate = (v / L) * np.tan(delta)
        np.testing.assert_allclose(result.yaw_rate, expected_yaw_rate, rtol=1e-10)


class TestBicycleUnicycleEquivalence:
    """Test that bicycle ≈ unicycle when yaw_rate = v/L * tan(delta)."""
    
    def test_equivalence_small_steering(self):
        """Test bicycle-unicycle equivalence for small steering angles."""
        v = 2.0  # m/s
        delta = 0.05  # rad (small angle)
        L = 2.7  # m
        dt = 0.1  # s
        N = 50
        
        # Simulate bicycle
        bicycle_result = vp.simulate_bicycle(
            v=np.full(N, v),
            delta=np.full(N, delta),
            L=L,
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Simulate equivalent unicycle
        yaw_rate = (v / L) * np.tan(delta)
        unicycle_result = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Trajectories should match closely
        np.testing.assert_allclose(bicycle_result.x, unicycle_result.x, rtol=1e-6)
        np.testing.assert_allclose(bicycle_result.y, unicycle_result.y, rtol=1e-6)
        np.testing.assert_allclose(bicycle_result.yaw, unicycle_result.yaw, rtol=1e-6)


class TestDataFrameWrappers:
    """Test DataFrame wrapper functions."""
    
    def test_simulate_from_dataframe_unicycle(self):
        """Test unicycle simulation from DataFrame."""
        df = pd.DataFrame({
            't': [0.0, 0.1, 0.2, 0.3],
            'v': [1.0, 1.0, 1.0, 1.0],
            'yaw_rate': [0.0, 0.0, 0.0, 0.0]
        })
        
        result = vp.simulate_from_dataframe_unicycle(df, init=vp.Pose(0, 0, 0))
        
        assert len(result.t) == 4
        assert result.x[0] == 0.0
        assert result.x[-1] > 0.0  # moved forward
    
    def test_simulate_from_dataframe_bicycle(self):
        """Test bicycle simulation from DataFrame."""
        df = pd.DataFrame({
            't': [0.0, 0.1, 0.2, 0.3],
            'v': [2.0, 2.0, 2.0, 2.0],
            'delta': [0.0, 0.0, 0.0, 0.0]
        })
        
        result = vp.simulate_from_dataframe_bicycle(df, L=2.7, init=vp.Pose(0, 0, 0))
        
        assert len(result.t) == 4
        assert result.x[0] == 0.0
        assert result.x[-1] > 0.0  # moved forward
    
    def test_missing_columns_unicycle(self):
        """Test error handling for missing columns in unicycle DataFrame."""
        df = pd.DataFrame({
            't': [0.0, 0.1],
            'v': [1.0, 1.0]
            # missing yaw_rate
        })
        
        with pytest.raises(ValueError, match="must have columns"):
            vp.simulate_from_dataframe_unicycle(df)
    
    def test_missing_columns_bicycle(self):
        """Test error handling for missing columns in bicycle DataFrame."""
        df = pd.DataFrame({
            't': [0.0, 0.1],
            'v': [1.0, 1.0]
            # missing delta
        })
        
        with pytest.raises(ValueError, match="must have columns"):
            vp.simulate_from_dataframe_bicycle(df, L=2.7)


class TestCSVHelpers:
    """Test CSV input/output helpers."""
    
    def test_csv_round_trip_unicycle(self):
        """Test saving and loading unicycle data."""
        # Create test data
        df_input = pd.DataFrame({
            't': [0.0, 0.1, 0.2],
            'v': [1.0, 1.5, 2.0],
            'yaw_rate': [0.0, 0.1, 0.2]
        })
        
        # Simulate
        result = vp.simulate_from_dataframe_unicycle(df_input, init=vp.Pose(0, 0, 0))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            vp.save_path_csv(result, temp_path)
            
            # Load back
            df_output = pd.read_csv(temp_path)
            
            # Check columns exist
            assert 't' in df_output.columns
            assert 'x' in df_output.columns
            assert 'y' in df_output.columns
            assert 'yaw' in df_output.columns
            assert 'v' in df_output.columns
            assert 'yaw_rate' in df_output.columns
            
            # Check values match
            np.testing.assert_allclose(df_output['t'].values, result.t)
            np.testing.assert_allclose(df_output['x'].values, result.x)
            np.testing.assert_allclose(df_output['y'].values, result.y)
        finally:
            os.unlink(temp_path)
    
    def test_read_inputs_csv(self):
        """Test reading input CSV."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
            f.write('t,v,yaw_rate\n')
            f.write('0.0,1.0,0.0\n')
            f.write('0.1,1.5,0.1\n')
        
        try:
            df = vp.read_inputs_csv(temp_path)
            
            assert len(df) == 2
            assert 't' in df.columns
            assert 'v' in df.columns
            assert 'yaw_rate' in df.columns
        finally:
            os.unlink(temp_path)


class TestENULLAConversions:
    """Test ENU ↔︎ LLA coordinate conversions."""
    
    def test_enu_to_lla_origin(self):
        """Test conversion at origin."""
        lat0, lon0 = 37.7749, -122.4194  # San Francisco
        
        lat, lon = vp.enu_to_lla(0.0, 0.0, lat0, lon0)
        
        assert lat == lat0
        assert lon == lon0
    
    def test_lla_to_enu_origin(self):
        """Test conversion at origin."""
        lat0, lon0 = 37.7749, -122.4194
        
        x, y = vp.lla_to_enu(lat0, lon0, lat0, lon0)
        
        np.testing.assert_allclose(x, 0.0, atol=1e-6)
        np.testing.assert_allclose(y, 0.0, atol=1e-6)
    
    def test_round_trip(self):
        """Test ENU -> LLA -> ENU round trip."""
        lat0, lon0 = 37.7749, -122.4194
        x_orig, y_orig = 100.0, 200.0
        
        # Convert to LLA
        lat, lon = vp.enu_to_lla(x_orig, y_orig, lat0, lon0)
        
        # Convert back to ENU
        x, y = vp.lla_to_enu(lat, lon, lat0, lon0)
        
        np.testing.assert_allclose(x, x_orig, rtol=1e-6)
        np.testing.assert_allclose(y, y_orig, rtol=1e-6)
    
    def test_arrays(self):
        """Test with array inputs."""
        lat0, lon0 = 37.7749, -122.4194
        x_arr = np.array([0.0, 100.0, 200.0])
        y_arr = np.array([0.0, 150.0, 300.0])
        
        lat_arr, lon_arr = vp.enu_to_lla(x_arr, y_arr, lat0, lon0)
        
        assert len(lat_arr) == 3
        assert len(lon_arr) == 3
        
        # Round trip
        x_back, y_back = vp.lla_to_enu(lat_arr, lon_arr, lat0, lon0)
        np.testing.assert_allclose(x_back, x_arr, rtol=1e-6)
        np.testing.assert_allclose(y_back, y_arr, rtol=1e-6)


class TestExampleGenerators:
    """Test example data generators."""
    
    def test_example_s_curve(self):
        """Test S-curve example generator."""
        df = vp.example_s_curve(duration=10.0, dt=0.1)
        
        assert 't' in df.columns
        assert 'v' in df.columns
        assert 'yaw_rate' in df.columns
        
        assert len(df) == 100  # 10.0 / 0.1
        assert df['t'].iloc[0] == 0.0
        assert df['t'].iloc[-1] < 10.0
    
    def test_example_slalom(self):
        """Test slalom example generator."""
        df = vp.example_slalom(duration=12.0, dt=0.05)
        
        assert 't' in df.columns
        assert 'v' in df.columns
        assert 'delta' in df.columns
        
        assert len(df) == 240  # 12.0 / 0.05
        assert df['t'].iloc[0] == 0.0
    
    def test_example_circle(self):
        """Test circle example generator."""
        # Test default 50m radius circle
        df = vp.example_circle(radius=50.0, v=2.0, dt=0.1, n_circles=1.0)
        
        assert 't' in df.columns
        assert 'v' in df.columns
        assert 'yaw_rate' in df.columns
        
        # Check that speed is constant
        np.testing.assert_allclose(df['v'].values, 2.0)
        
        # Check that yaw_rate is constant and equals v/radius
        expected_yaw_rate = 2.0 / 50.0
        np.testing.assert_allclose(df['yaw_rate'].values, expected_yaw_rate)
        
        # Simulate and verify radius
        path = vp.simulate_from_dataframe_unicycle(df, init=vp.Pose(0, 0, 0))
        
        # For a circle starting at (0,0) with yaw=0, center is at (0, radius)
        center_x = 0.0
        center_y = 50.0
        distances = np.sqrt((path.x - center_x)**2 + (path.y - center_y)**2)
        
        # Check that all points are approximately at radius distance from center
        np.testing.assert_allclose(distances, 50.0, rtol=1e-3)
    
    def test_example_circle_custom_radius(self):
        """Test circle with custom radius."""
        radius = 100.0
        df = vp.example_circle(radius=radius, v=3.0, dt=0.1, n_circles=0.5)
        
        # Check yaw_rate calculation
        expected_yaw_rate = 3.0 / radius
        np.testing.assert_allclose(df['yaw_rate'].values, expected_yaw_rate)


class TestErrorHandling:
    """Test error handling."""
    
    def test_length_mismatch_unicycle(self):
        """Test error for mismatched input lengths in unicycle."""
        v = np.array([1.0, 2.0, 3.0])
        yaw_rate = np.array([0.0, 0.1])  # different length
        dt = 0.1
        
        with pytest.raises(ValueError, match="Length mismatch"):
            vp.simulate_unicycle(v=v, yaw_rate=yaw_rate, dt=dt)
    
    def test_length_mismatch_bicycle(self):
        """Test error for mismatched input lengths in bicycle."""
        v = np.array([1.0, 2.0, 3.0])
        delta = np.array([0.0, 0.1])  # different length
        L = 2.7
        dt = 0.1
        
        with pytest.raises(ValueError, match="Length mismatch"):
            vp.simulate_bicycle(v=v, delta=delta, L=L, dt=dt)
    
    def test_no_dt_or_t(self):
        """Test error when neither dt nor t is provided."""
        v = np.array([1.0, 2.0])
        yaw_rate = np.array([0.0, 0.1])
        
        with pytest.raises(ValueError, match="Must provide either dt or t"):
            vp.simulate_unicycle(v=v, yaw_rate=yaw_rate)


class TestNoiseInjection:
    """Test noise injection functionality."""
    
    def test_unicycle_with_noise(self):
        """Test unicycle simulation with noise."""
        v = 1.0
        yaw_rate = 0.0
        dt = 0.1
        N = 100
        
        # With noise, results should be different from noiseless
        rng = np.random.default_rng(42)
        result_noisy = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, 0),
            noise_std_v=0.1,
            noise_std_yawrate=0.01,
            rng=rng
        )
        
        result_clean = vp.simulate_unicycle(
            v=np.full(N, v),
            yaw_rate=np.full(N, yaw_rate),
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Results should differ due to noise
        assert not np.allclose(result_noisy.x, result_clean.x)
    
    def test_bicycle_with_noise(self):
        """Test bicycle simulation with noise."""
        v = 2.0
        delta = 0.0
        L = 2.7
        dt = 0.1
        N = 100
        
        # With noise, results should be different from noiseless
        rng = np.random.default_rng(42)
        result_noisy = vp.simulate_bicycle(
            v=np.full(N, v),
            delta=np.full(N, delta),
            L=L,
            dt=dt,
            init=vp.Pose(0, 0, 0),
            noise_std_v=0.1,
            noise_std_delta=0.01,
            rng=rng
        )
        
        result_clean = vp.simulate_bicycle(
            v=np.full(N, v),
            delta=np.full(N, delta),
            L=L,
            dt=dt,
            init=vp.Pose(0, 0, 0)
        )
        
        # Results should differ due to noise
        assert not np.allclose(result_noisy.x, result_clean.x)


class TestTimestampModes:
    """Test different timestamp modes."""
    
    def test_fixed_dt_scalar(self):
        """Test with scalar dt."""
        result = vp.simulate_unicycle(
            v=1.0,
            yaw_rate=0.0,
            dt=0.1,
            init=vp.Pose(0, 0, 0)
        )
        
        # Should create at least one timestep
        assert len(result.t) > 0
    
    def test_timestamp_array(self):
        """Test with timestamp array."""
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        result = vp.simulate_unicycle(
            v=1.0,
            yaw_rate=0.0,
            t=t,
            init=vp.Pose(0, 0, 0)
        )
        
        np.testing.assert_array_equal(result.t, t)
        assert len(result.x) == len(t)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
