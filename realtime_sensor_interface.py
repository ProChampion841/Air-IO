"""
Real IMU Sensor Interface
Supports serial communication with IMU sensors
"""
import serial
import numpy as np
import time
from collections import deque


class RealIMUSensor:
    """Interface for real IMU sensor via serial port"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, imu_rate=200):
        """
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows)
            baudrate: Serial baudrate
            imu_rate: IMU sampling rate in Hz
        """
        self.port = port
        self.baudrate = baudrate
        self.dt = 1.0 / imu_rate
        self.serial = None
        self.last_timestamp = None
        
    def connect(self):
        """Connect to IMU sensor"""
        try:
            self.serial = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)  # Wait for connection to stabilize
            
            # Flush initial data
            self.serial.flushInput()
            self.serial.flushOutput()
            
            print(f"✓ Connected to IMU on {self.port} @ {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def read(self):
        """Read one IMU sample
        
        Expected format: "timestamp,gx,gy,gz,ax,ay,az\\n"
        
        Returns:
            dict with keys: 'acc', 'gyro', 'dt', 'timestamp'
            or None if read failed
        """
        if self.serial is None or not self.serial.is_open:
            raise RuntimeError("Sensor not connected. Call connect() first.")
        
        try:
            # Read line from serial
            line = self.serial.readline().decode('utf-8').strip()
            
            if not line or line.startswith('#'):
                return None
            
            # Parse: timestamp,gx,gy,gz,ax,ay,az
            data = [float(x) for x in line.split(',')]
            
            if len(data) != 7:
                return None
            
            timestamp = data[0]
            gyro = np.array(data[1:4], dtype=np.float32)  # rad/s
            acc = np.array(data[4:7], dtype=np.float32)   # m/s^2
            
            # Calculate dt
            if self.last_timestamp is not None:
                dt = timestamp - self.last_timestamp
            else:
                dt = self.dt
            
            self.last_timestamp = timestamp
            
            return {
                'acc': acc,
                'gyro': gyro,
                'dt': dt,
                'timestamp': timestamp
            }
            
        except (ValueError, UnicodeDecodeError, IndexError) as e:
            # Skip malformed data
            return None
    
    def disconnect(self):
        """Disconnect from sensor"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("✓ Disconnected from IMU")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class IMUDataLogger:
    """Log IMU data to file for later processing"""
    
    def __init__(self, filename='imu_log.csv'):
        self.filename = filename
        self.file = None
        
    def start(self):
        """Start logging"""
        self.file = open(self.filename, 'w')
        self.file.write("# timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z\n")
        print(f"✓ Logging to {self.filename}")
        
    def log(self, imu_data):
        """Log one IMU sample"""
        if self.file:
            self.file.write(f"{imu_data['timestamp']:.6f},")
            self.file.write(f"{imu_data['gyro'][0]:.6f},{imu_data['gyro'][1]:.6f},{imu_data['gyro'][2]:.6f},")
            self.file.write(f"{imu_data['acc'][0]:.6f},{imu_data['acc'][1]:.6f},{imu_data['acc'][2]:.6f}\n")
            
    def stop(self):
        """Stop logging"""
        if self.file:
            self.file.close()
            print(f"✓ Log saved to {self.filename}")


def test_sensor_connection(port='/dev/ttyUSB0', baudrate=115200, duration=5.0):
    """Test IMU sensor connection and display data"""
    print("="*70)
    print("IMU Sensor Connection Test")
    print("="*70)
    
    sensor = RealIMUSensor(port=port, baudrate=baudrate)
    
    if not sensor.connect():
        return False
    
    print(f"\nReading data for {duration} seconds...")
    print("Format: timestamp | gyro (rad/s) | acc (m/s^2)\n")
    
    start_time = time.time()
    sample_count = 0
    
    try:
        while (time.time() - start_time) < duration:
            data = sensor.read()
            
            if data is not None:
                sample_count += 1
                
                if sample_count % 20 == 0:  # Display every 0.1s @ 200Hz
                    print(f"[{data['timestamp']:10.3f}] "
                          f"Gyro: [{data['gyro'][0]:7.4f}, {data['gyro'][1]:7.4f}, {data['gyro'][2]:7.4f}] "
                          f"Acc: [{data['acc'][0]:7.4f}, {data['acc'][1]:7.4f}, {data['acc'][2]:7.4f}]")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        sensor.disconnect()
    
    elapsed = time.time() - start_time
    rate = sample_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*70)
    print(f"Received {sample_count} samples in {elapsed:.2f}s ({rate:.1f} Hz)")
    print("="*70)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test IMU sensor connection')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                        help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Test duration in seconds')
    
    args = parser.parse_args()
    
    test_sensor_connection(args.port, args.baudrate, args.duration)
