import numpy as np
import pyproj

def gps_to_xyz(lat, lon, alt, origin_lat=None, origin_lon=None, origin_alt=None, frame='ENU'):
    """
    Convert GPS (lat, lon, alt) to local coordinates (x, y, z)
    
    Args:
        lat, lon, alt: GPS coordinates (arrays or scalars)
        origin_lat, origin_lon, origin_alt: Reference point (uses first point if None)
        frame: 'ENU' (East-North-Up) or 'NED' (North-East-Down)
    
    Returns:
        x, y, z: Local coordinates in meters
    """
    # Use first point as origin if not specified
    if origin_lat is None:
        origin_lat = lat[0] if hasattr(lat, '__len__') else lat
    if origin_lon is None:
        origin_lon = lon[0] if hasattr(lon, '__len__') else lon
    if origin_alt is None:
        origin_alt = alt[0] if hasattr(alt, '__len__') else alt
    
    # Define coordinate systems
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    # Convert origin to ECEF
    origin_x, origin_y, origin_z = pyproj.transform(lla, ecef, origin_lon, origin_lat, origin_alt)
    
    # Convert points to ECEF
    x_ecef, y_ecef, z_ecef = pyproj.transform(lla, ecef, lon, lat, alt)
    
    # Calculate ENU transformation matrix
    lat_rad = np.radians(origin_lat)
    lon_rad = np.radians(origin_lon)
    
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]
    ])
    
    # Transform to ENU
    ecef_diff = np.array([x_ecef - origin_x, y_ecef - origin_y, z_ecef - origin_z])
    enu = R @ ecef_diff
    
    if frame.upper() == 'NED':
        # Convert ENU to NED: [N, E, D] = [E_y, E_x, -E_z]
        return enu[1], enu[0], -enu[2]  # x (North), y (East), z (Down)
    else:
        # ENU frame
        return enu[0], enu[1], enu[2]  # x (East), y (North), z (Up)


# Example usage
if __name__ == "__main__":
    # Sample GPS data
    lat = np.array([40.7128, 40.7129, 40.7130])
    lon = np.array([-74.0060, -74.0059, -74.0058])
    alt = np.array([10.0, 10.5, 11.0])
    
    x, y, z = gps_to_xyz(lat, lon, alt)
    
    print("GPS to XYZ conversion:")
    for i in range(len(lat)):
        print(f"GPS: ({lat[i]:.6f}, {lon[i]:.6f}, {alt[i]:.2f}) -> XYZ: ({x[i]:.3f}, {y[i]:.3f}, {z[i]:.3f})")
