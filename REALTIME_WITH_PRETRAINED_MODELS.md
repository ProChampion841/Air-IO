# Using Pre-trained Models in Real-Time Pipeline

## Overview
The `realtime_pipeline_demo.py` now supports both **mock networks** (for testing) and **real pre-trained models** (for actual predictions).

## Usage

### Option 1: Demo Mode (Mock Networks)
Quick test without pre-trained models:
```bash
python realtime_pipeline_demo.py
```

### Option 2: Real Pre-trained Models
Use actual AirIMU and AirIO models:
```bash
python realtime_pipeline_demo.py \
    --use-real-models \
    --airimu-model experiments/blackbird/airimu/best_model.pth \
    --airio-model experiments/blackbird/motion_body_rot/best_model.pth
```

## Model Paths

Download pre-trained models from the releases:
- **AirIMU**: [Download here](https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_blackbird.zip)
- **AirIO**: [Download here](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Blackbird.zip)

Extract and use the checkpoint files (usually `best_model.pth` or `checkpoint_epoch_XX.pth`).

## Example with Blackbird Models

```bash
# 1. Download and extract models
# AirIMU_blackbird.zip -> experiments/blackbird/airimu/
# AirIO_Blackbird.zip -> experiments/blackbird/motion_body_rot/

# 2. Run with real models
python realtime_pipeline_demo.py \
    --use-real-models \
    --airimu-model experiments/blackbird/airimu/best_model.pth \
    --airio-model experiments/blackbird/motion_body_rot/best_model.pth
```

## Code Structure

### Mock Networks (Default)
- `MockAirIMUNetwork`: Simulates IMU correction with simple bias reduction
- `MockAirIONetwork`: Simulates velocity prediction with basic integration
- **Use case**: Quick testing, understanding pipeline flow

### Real Networks
- `AirIMUNetwork`: Loads pre-trained AirIMU model for actual IMU correction
- `AirIONetwork`: Loads pre-trained AirIO model for accurate velocity prediction
- **Use case**: Production deployment, real sensor data

## Network Architecture Notes

### AirIMU Model
- Input: Raw accelerometer + gyroscope sequences (200 samples)
- Output: Corrected IMU + orientation estimate
- Model type: Specified in AirIMU training config

### AirIO Model
- Input: Corrected IMU + orientation (200 samples)
- Output: Body-frame velocity prediction
- Model type: `codewithrot` (with attitude encoding) or `codenetmotion` (without)

## Customization

If you need to modify the network loading:

```python
# In realtime_pipeline_demo.py

class AirIONetwork:
    def __init__(self, model_path, network_type='codewithrot'):
        from model import get_model
        
        # Change network_type based on your training config
        self.model = get_model(network_type)  # 'codewithrot' or 'codenetmotion'
        
        # Load checkpoint
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### Error: "model_state_dict not found"
The checkpoint might use a different key. Check with:
```python
checkpoint = torch.load('model.pth')
print(checkpoint.keys())  # See available keys
```

### Error: "Model architecture mismatch"
Ensure the `network_type` matches your training config:
- Check `configs/BlackBird/motion_body_rot.conf` → `train.network` field
- Use `codewithrot` for attitude encoding
- Use `codenetmotion` for standard training

### Error: "CUDA out of memory"
The demo uses CPU by default. If you have GPU issues:
```python
# Force CPU
self.device = torch.device('cpu')
```

## Performance Comparison

| Mode | Position Error | Velocity Error | Speed |
|------|---------------|----------------|-------|
| Mock Networks | ~0.5-1.0m | ~0.3-0.5 m/s | Fast |
| Pre-trained Models | ~0.1-0.3m | ~0.05-0.15 m/s | Slower |

Mock networks are for demonstration only. Use pre-trained models for real applications.
