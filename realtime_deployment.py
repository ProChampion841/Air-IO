"""
Real-Time IMU Odometry Deployment
Run AirIO pipeline with real IMU sensor data
"""
import os
import sys
import torch
import numpy as np
import pypose as pp
import time
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from realtime_sensor_interface import RealIMUSensor, IMUDataLogger
from realtime_pipeline_demo import RealtimeOdometryPipeline


def run_realtime_with_sensor(
    sensor_port='/dev/ttyUSB0',
    baudrate=115200,
    airimu_model=None,
    airio_model=None,
    duration=60.0,
    save_results=True,
    log_imu=False,
    display_rate=10  # Hz
):
    """Run real-time odometry with real IMU sensor
    
    Args:
        sensor_port: Serial port for IMU
        baudrate: Serial baudrate
        airimu_model: Path to AirIMU model (optional)
        airio_model: Path to AirIO model (required)
        duration: Duration in seconds
        save_results: Save odometry results to file
        log_imu: Log raw IMU data to file
        display_rate: Display update rate in Hz
    """
    
    print("="*70)
    print("Real-Time IMU Odometry with Real Sensor")
    print("="*70)
    print(f"Sensor: {sensor_port} @ {baudrate} baud")
    print(f"Duration: {duration}s")
    print(f"Models: AirIMU={'Yes' if airimu_model else 'No'}, AirIO={'Yes' if airio_model else 'No'}")
    print("="*70)
    
    # Initialize sensor
    print("\n[1/4] Connecting to IMU sensor...")
    sensor = RealIMUSensor(port=sensor_port, baudrate=baudrate, imu_rate=200)
    if not sensor.connect():
        print("Failed to connect to sensor. Exiting.")
        return
    
    # Initialize logger
    logger = None
    if log_imu:
        logger = IMUDataLogger(f'imu_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        logger.start()
    
    # Initialize pipeline
    print("\n[2/4] Loading models...")
    use_mock = (airio_model is None)
    if use_mock:
        print("WARNING: No models provided, using mock networks (demo only)")
    
    pipeline = RealtimeOdometryPipeline(
        window_size=200,
        airimu_model_path=airimu_model,
        airio_model_path=airio_model,
        use_mock=use_mock
    )
    
    # Initialize state (stationary start)
    print("\n[3/4] Initializing state...")
    print("Assuming stationary start at origin")
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_rot = pp.identity_SO3(1)
    pipeline.initialize(init_pos, init_rot, init_vel)
    
    # Run real-time processing
    print("\n[4/4] Processing IMU data in real-time...")
    print("Press Ctrl+C to stop\n")
    print(f"{'Time':>8} | {'Position (m)':^30} | {'Velocity (m/s)':^30}")
    print("-"*70)
    
    start_time = time.time()
    sample_count = 0
    results = []
    display_interval = 1.0 / display_rate
    last_display_time = 0
    
    try:
        while (time.time() - start_time) < duration:
            # Read IMU
            imu_data = sensor.read()
            if imu_data is None:
                continue
            
            # Log raw data
            if logger:
                logger.log(imu_data)
            
            # Process through pipeline
            result = pipeline.process_imu(imu_data)
            
            sample_count += 1
            current_time = time.time() - start_time
            
            # Display at specified rate
            if current_time - last_display_time >= display_interval:
                pos = result['position']
                vel = result['velocity']
                print(f"{current_time:7.2f}s | "
                      f"[{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}] | "
                      f"[{vel[0]:7.3f}, {vel[1]:7.3f}, {vel[2]:7.3f}]")
                last_display_time = current_time
            
            # Save results
            if save_results:
                results.append({
                    'timestamp': imu_data['timestamp'],
                    'position': result['position'].copy(),
                    'velocity': result['velocity'].copy(),
                    'orientation': result['orientation'].matrix().numpy(),
                    'gyro_bias': pipeline.state[9:12].numpy().copy(),
                    'acc_bias': pipeline.state[12:15].numpy().copy()
                })
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        sensor.disconnect()
        if logger:
            logger.stop()
    
    # Save results
    elapsed = time.time() - start_time
    
    if save_results and len(results) > 0:
        output_file = f"realtime_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(output_file, results)
        print(f"\n✓ Odometry results saved to {output_file}")
    
    # Statistics
    print("\n" + "="*70)
    print("Statistics:")
    print(f"  Samples processed: {sample_count}")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Average rate: {sample_count/elapsed:.1f} Hz")
    
    if len(results) > 0:
        final_pos = results[-1]['position']
        final_vel = results[-1]['velocity']
        print(f"  Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
        print(f"  Final velocity: [{final_vel[0]:.3f}, {final_vel[1]:.3f}, {final_vel[2]:.3f}] m/s")
        
        # Estimated biases
        gyro_bias = results[-1]['gyro_bias']
        acc_bias = results[-1]['acc_bias']
        print(f"  Gyro bias: [{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}] rad/s")
        print(f"  Acc bias: [{acc_bias[0]:.6f}, {acc_bias[1]:.6f}, {acc_bias[2]:.6f}] m/s²")
    
    print("="*70)


def visualize_results(results_file):
    """Visualize saved odometry results"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print(f"Loading results from {results_file}...")
    results = np.load(results_file, allow_pickle=True)
    
    timestamps = np.array([r['timestamp'] for r in results])
    positions = np.array([r['position'] for r in results])
    velocities = np.array([r['velocity'] for r in results])
    
    # Normalize timestamps
    timestamps = timestamps - timestamps[0]
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2D trajectory (top view)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Position vs time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(timestamps, positions[:, 0], label='X')
    ax3.plot(timestamps, positions[:, 1], label='Y')
    ax3.plot(timestamps, positions[:, 2], label='Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # Velocity vs time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(timestamps, velocities[:, 0], label='Vx')
    ax4.plot(timestamps, velocities[:, 1], label='Vy')
    ax4.plot(timestamps, velocities[:, 2], label='Vz')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity vs Time')
    ax4.legend()
    ax4.grid(True)
    
    # Speed
    ax5 = fig.add_subplot(2, 3, 5)
    speed = np.linalg.norm(velocities, axis=1)
    ax5.plot(timestamps, speed, 'b-')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed vs Time')
    ax5.grid(True)
    
    # Distance traveled
    ax6 = fig.add_subplot(2, 3, 6)
    distances = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)
    ax6.plot(timestamps, distances, 'b-')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance (m)')
    ax6.set_title(f'Total Distance: {distances[-1]:.2f} m')
    ax6.grid(True)
    
    plt.tight_layout()
    
    output_file = results_file.replace('.npy', '_plot.png')
    plt.savefig(output_file, dpi=150)
    print(f"✓ Plot saved to {output_file}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time IMU odometry deployment')
    
    # Sensor settings
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                        help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate')
    
    # Model settings
    parser.add_argument('--airimu-model', type=str, default=None,
                        help='Path to AirIMU model checkpoint')
    parser.add_argument('--airio-model', type=str, default=None,
                        help='Path to AirIO model checkpoint')
    
    # Runtime settings
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duration in seconds (0 = infinite)')
    parser.add_argument('--save', action='store_true',
                        help='Save odometry results to file')
    parser.add_argument('--log-imu', action='store_true',
                        help='Log raw IMU data to file')
    parser.add_argument('--display-rate', type=float, default=10.0,
                        help='Display update rate in Hz')
    
    # Visualization
    parser.add_argument('--visualize', type=str, default=None,
                        help='Visualize saved results file')
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_results(args.visualize)
    else:
        run_realtime_with_sensor(
            sensor_port=args.port,
            baudrate=args.baudrate,
            airimu_model=args.airimu_model,
            airio_model=args.airio_model,
            duration=args.duration if args.duration > 0 else float('inf'),
            save_results=args.save,
            log_imu=args.log_imu,
            display_rate=args.display_rate
        )
