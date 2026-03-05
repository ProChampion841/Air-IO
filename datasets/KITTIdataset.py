"""
KITTI dataset class for Air-IO (using Blackbird format)
"""
import os
import numpy as np
import pypose as pp
import torch
from utils import qinterp
from .dataset import Sequence
import pickle

class KITTI(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81, 
        remove_g=False,
        **kwargs
    ):
        super(KITTI, self).__init__()
        (
            self.data_root,
            self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)
        data_path = os.path.join(data_root, data_name)
        
        # Load data
        self.load_imu(data_path)
        self.load_gt(data_path)
        
        # Process data
        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)
        
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])
        
        # Set orientation
        self.set_orientation(rot_path, data_name, rot_type)
        
        # Transform to global/body frame
        self.update_coordinate(coordinate, mode)
        
        # Remove gravity term
        self.remove_gravity(remove_g)
        
    def get_length(self):
        return self.data["time"].shape[0]

    def load_imu(self, folder):
        """Load IMU data from Blackbird format CSV"""
        imu_file = os.path.join(folder, "imu_data.csv")
        imu_data = np.loadtxt(imu_file, dtype=float, delimiter=",")
        
        self.data["time"] = imu_data[:, 0] / 1e6  # Convert from microseconds to seconds
        self.data["gyro"] = imu_data[:, 1:4]  # wx, wy, wz
        self.data["acc"] = imu_data[:, 4:]    # ax, ay, az
    
    def load_gt(self, folder):
        """Load ground truth poses from Blackbird format CSV"""
        gt_file = os.path.join(folder, "groundTruthPoses.csv")
        gt_data = np.loadtxt(gt_file, dtype=float, delimiter=",")
        
        self.data["gt_time"] = gt_data[:, 0] / 1e6  # Convert from microseconds to seconds
        self.data["pos"] = gt_data[:, 1:4]  # x, y, z
        quat = gt_data[:, 4:8]  # qw, qx, qy, qz
        
        # Convert to PyPose SO3 format (xyzw)
        quat_xyzw = np.zeros_like(quat)
        quat_xyzw[:, 0:3] = quat[:, 1:4]  # qx, qy, qz
        quat_xyzw[:, 3] = quat[:, 0]      # qw
        
        # Interpolate to IMU timestamps
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], quat_xyzw
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )
        
        # Compute velocity from position
        self.data["velocity"] = self.compute_velocity(
            self.data["gt_translation"], self.data["time"]
        )

    def interp_rot(self, time, opt_time, quat):
        """Interpolate rotations using SLERP"""
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double()
        return pp.SO3(quat)

    def interp_xyz(self, time, opt_time, xyz):
        """Interpolate positions"""
        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)
    
    def compute_velocity(self, positions, times):
        """Compute velocity from positions using finite differences"""
        vel = torch.zeros_like(positions)
        # Forward difference for first point
        vel[0] = (positions[1] - positions[0]) / (times[1] - times[0])
        # Central difference for middle points
        for i in range(1, len(positions) - 1):
            vel[i] = (positions[i+1] - positions[i-1]) / (times[i+1] - times[i-1])
        # Backward difference for last point
        vel[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
        
        # Smooth velocity with moving average
        window = 5
        for i in range(window, len(vel) - window):
            vel[i] = vel[i-window:i+window+1].mean(dim=0)
        
        return vel

    def update_coordinate(self, coordinate, mode):
        """Transform data based on coordinate frame"""
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise e

    def set_orientation(self, exp_path, data_name, rotation_type):
        """Set orientation from external source if provided"""
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)
            state = loaded_data[data_name]
            
            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                print(f"Unsupported rotation type: {rotation_type}")
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {exp_path} was not found.")
            raise
    
    def remove_gravity(self, remove_g):
        """Remove gravity from accelerometer data"""
        if remove_g is True:
            print("gravity has been removed")
            self.data["acc"] -= self.g_vector
