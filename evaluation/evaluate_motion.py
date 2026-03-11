# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import pypose as pp

import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import imu_seq_collate,SeqDataset
 
from utils import CPU_Unpickler, integrate, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos

from utils.visualize_state import visualize_motion
 
def calculate_rte(outstate,duration, step_size):
    poses, poses_gt = outstate['poses'],outstate['poses_gt'][1:,:]

    dp = poses[:, duration-1:] - poses[:, :-duration+1]
    dp_gt = poses_gt[duration-1:] - poses_gt[:-duration+1]
    rte = (dp - dp_gt).norm(dim=-1)  
    return rte
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/euroc/motion_body", help="Path for AirIO netoutput")
    parser.add_argument("--seqlen", type=int, default="1000", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir",type=str,default = "./result/loss_result",help = "Directory where the results wiil be saved")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")

    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    print(dataset_conf.keys())


    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")
    
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)

    AllResults = []
    net_out_result = {}

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            print("data_conf.dataroot", data_conf.data_root)
            print("data_name", data_name)
            print("data_conf.name", data_conf.name)

            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            
            integrator_reset = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity = gravity,
                reset=True
            ).to(args.device).double()
            
            outstate = integrate(
                integrator_outstate, loader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=args.usegtrot
            )
            relative_outstate = integrate(
                integrator_reset, loader, init, 
                device=args.device, gtinit=True,
                use_gt_rot=args.usegtrot
            )
            
            if args.exp is not None:
                motion_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
                motion_loader = Data.DataLoader(dataset=motion_dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            
                # Try to find matching key in pickle file
                if data_name not in inference_state_load:
                    print(f"[WARN] Key '{data_name}' not found in pickle file.")
                    print(f"Available keys: {list(inference_state_load.keys())}")
                    # Try alternative key formats
                    alt_key = data_name.replace('/', '_')
                    if alt_key in inference_state_load:
                        print(f"Using alternative key: {alt_key}")
                        data_name = alt_key
                    else:
                        raise KeyError(f"Cannot find matching key for {data_name}")
                
                inference_state = inference_state_load[data_name] 
                gt_ts =  motion_dataset.data['time']
                vel_ts = inference_state['ts']
                indices = torch.cat([torch.where(gt_ts == item)[0] for item in vel_ts[:,0]]).to(torch.int32)

                if "coordinate" in dataset_conf.keys():
                    print("*************",dataset_conf["coordinate"],"*************")
                    if dataset_conf["coordinate"] == "body_coord":
                        rotation = motion_dataset.data['gt_orientation'] # Set data['gt_orientation'] using AirIMU or gt-truth in the dataconf

                        vel_dist = rotation[indices,:] * inference_state['net_vel'] - motion_dataset.data['velocity'][indices,:]  
                        net_vel = interp_xyz(gt_ts, vel_ts[:,0],  inference_state['net_vel'])
                        net_vel = rotation * net_vel
                    
                    if dataset_conf["coordinate"] == "glob_coord":
                        vel_dist = inference_state['net_vel'] - motion_dataset.data['velocity'][indices,:]    
                        net_vel = interp_xyz(gt_ts, vel_ts[:,0], inference_state['net_vel'])

                if data_conf.name == "BlackBird":
                    save_prefix = os.path.dirname(data_name).split('/')[1]
                else:
                    save_prefix = data_name
               
                dt = gt_ts[1:] - gt_ts[:-1]
                data_inte = {"vel":net_vel,'dt':dt}
                
                integrator_vel =Velocity_Integrator(
                    init['pos']).to(args.device).double()
                
                inf_outstate =integrate_pos(
                    integrator_vel, data_inte, init, motion_dataset,
                    device=args.device
                )
                inf_rte = calculate_rte(inf_outstate, args.seqlen,args.seqlen)

                # Calculate position metrics (ATE)
                pos_error = inf_outstate['pos_dist']
                ate = torch.sqrt((pos_error**2).mean()).item()  # ATE is RMSE of position error
                pos_rmse = ate
                pos_max = pos_error.max().item()
                
                # Calculate velocity norm metrics
                vel_pred = inf_outstate['vel'] if 'vel' in inf_outstate else net_vel
                vel_gt = motion_dataset.data['velocity']
                vel_norm_error = (vel_pred.norm(dim=-1) - vel_gt.norm(dim=-1)).abs()
                vel_norm_rmse = torch.sqrt((vel_norm_error**2).mean()).item()
                vel_norm_max = vel_norm_error.max().item()
                
                # Calculate velocity vector metrics
                vel_error = inf_outstate['vel_dist'] if 'vel_dist' in inf_outstate else (vel_pred - vel_gt).norm(dim=-1)
                vel_rmse = torch.sqrt((vel_error**2).mean()).item()
                vel_max = vel_error.max().item()
                
                # Calculate angle error metrics
                # Note: Motion model only predicts velocity, rotation comes from dataset
                # If using ground truth rotation, angle error will be 0
                # To get meaningful angle error, use AirIMU or integration rotation in dataset config
                if 'rot_type' in dataset_conf and dataset_conf.rot_type is not None and dataset_conf.rot_type.lower() != 'none':
                    # Using predicted rotation (AirIMU or integration)
                    gt_rot = motion_dataset.data['gt_orientation']
                    pred_rot = rotation  # rotation from AirIMU or integration
                    angle_error = (pred_rot.Inv() @ gt_rot).Log().norm(dim=-1) * 180 / np.pi
                    angle_rmse = torch.sqrt((angle_error**2).mean()).item()
                    angle_max = angle_error.max().item()
                else:
                    # Using ground truth rotation, no angle error
                    angle_rmse = angle_max = 0.0
                
                # Calculate total flight distance
                gt_poses = outstate['poses_gt'][0]
                flight_dist = (gt_poses[1:] - gt_poses[:-1]).norm(dim=-1).sum().item()
                
                metrics = {
                    'ATE': ate,
                    'pos_rmse': pos_rmse, 'pos_max': pos_max,
                    'vel_rmse': vel_rmse, 'vel_max': vel_max,
                    'vel_norm_rmse': vel_norm_rmse, 'vel_norm_max': vel_norm_max,
                    'angle_rmse': angle_rmse, 'angle_max': angle_max,
                    'flight_dist': flight_dist
                }

                #save loss result
                result_dic = {
                    'name': data_name,      
                    'ATE':pos_rmse,
                    'AVE':inf_outstate['vel_dist'].mean().item(),
                    'RP_RMSE': np.sqrt((inf_rte**2).mean()).numpy().item(),
                    **metrics
                    }
                
                AllResults.append(result_dic)
                
                print("==============Integration==============")
                print("outstate:")
                print("pos_err: ", outstate['pos_dist'].mean())
                print("rte",relative_outstate['vel_dist'].mean())
                
                print("==============AirIO==============")
                print("infstate:")
                print(f"ATE: {metrics['ATE']:.4f} m")
                print(f"Position RMSE: {metrics['pos_rmse']:.4f} m, Max: {metrics['pos_max']:.4f} m")
                print(f"Velocity RMSE: {metrics['vel_rmse']:.4f} m/s, Max: {metrics['vel_max']:.4f} m/s")
                print(f"Velocity Norm RMSE: {metrics['vel_norm_rmse']:.4f} m/s, Max: {metrics['vel_norm_max']:.4f} m/s")
                print(f"Angle RMSE: {metrics['angle_rmse']:.4f} deg, Max: {metrics['angle_max']:.4f} deg")
                print(f"RTE: {inf_rte.mean():.4f} m")

            visualize_motion(save_prefix, folder,outstate,inf_outstate,metrics)          
            # visualize_motion(save_prefix, folder,outstate,inf_outstate,metrics)

        file_path = os.path.join(folder, "result.json")
        with open(file_path, 'w') as f: 
            json.dump(AllResults, f, indent=4)
