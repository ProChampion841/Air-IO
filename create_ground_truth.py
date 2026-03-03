import numpy as np
import pandas as pd
from gps_to_xyz import gps_to_xyz
from gps_orientation_converter import gps_orientation_to_quaternion

def create_ground_truth_csv(timestamps, lat, lon, alt, qx, qy, qz, qw, output_file, frame='ENU'):
    """
    Create groundTruthPoses.csv from GPS data
    
    Args:
        timestamps: Unix timestamps in nanoseconds
        lat, lon, alt: GPS coordinates
        qx, qy, qz, qw: Quaternion orientation (if you don't have this, use [0,0,0,1])
        output_file: Output CSV file path
        frame: 'ENU' or 'NED' coordinate frame
    """
    # Convert GPS to local XYZ
    x, y, z = gps_to_xyz(lat, lon, alt, frame=frame)
    
    # Create dataframe
    data = np.column_stack([timestamps, x, y, z, qx, qy, qz, qw])
    
    # Save to CSV (no header, comma-separated)
    np.savetxt(output_file, data, delimiter=',', 
               fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f'])
    
    print(f"Saved ground truth poses to {output_file}")


# Example: Load your GPS data and create CSV
if __name__ == "__main__":
    # Replace with your actual data loading
    # Example: df = pd.read_csv('your_gps_data.csv')
    
    timestamps = np.array([1525745925000109, 1525745925002855, 1525745925005633])
    lat = np.array([40.7128, 40.7129, 40.7130])
    lon = np.array([-74.0060, -74.0059, -74.0058])
    alt = np.array([10.0, 10.5, 11.0])
    
    # OPTION 1: If you have GPS orientation (roll, pitch, yaw) in degrees
    # orientation = np.column_stack([roll, pitch, yaw])  # Load your data
    # qx, qy, qz, qw = gps_orientation_to_quaternion(orientation, format='euler_deg')
    
    # OPTION 2: If you have GPS heading/course only (in degrees)
    heading = np.array([45, 90, 135])  # Replace with your heading data
    qx, qy, qz, qw = gps_orientation_to_quaternion(heading, format='heading_deg')
    
    create_ground_truth_csv(timestamps, lat, lon, alt, qx, qy, qz, qw, 
                           'groundTruthPoses.csv', frame='ENU')  # or frame='NED'
