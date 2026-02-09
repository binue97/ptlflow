# Docker Setup for PTLFlow

## Quick Start

```bash
cd docker

# Build
docker compose build

# Run interactive shell
docker compose run --rm ptlflow
```

## Training
```bash
# Set config

# Train with config
python train.py --config ${CONFIG_PATH}
```


## Stack

| Component | Version |
|-----------|---------|
| CUDA | 12.1.1 |
| cuDNN | 8 |
| Python | 3.12 |
| PyTorch | 2.5.1 |
| Ubuntu | 22.04 |

## Volume Mounts

| Host | Container |
|------|-----------|
| `/home/bnu/workspace/ptlflow` | `/workspace/ptlflow` |
| `/home/bnu/hdd/data` | `/data` |

## Troubleshooting

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Out of memory

Edit `docker-compose.yml`:
```yaml
shm_size: '16gb'
```
