"""
Modal Launcher for DDP Training
================================
Professional, clean interface for multi-GPU DDP training.
"""
import modal
from pathlib import Path
import os

# Create optimized image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",              # PyTorch 2.1+ with DDP improvements
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",      # 8-bit optimizer to save memory
    )
)

app = modal.App("tinyzero-ddp", image=image)
volume = modal.Volume.from_name("tinyzero-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("vstar-cache", create_if_missing=True)


@app.function(
    gpu="H100:4",                      # 4× H100 GPUs for DDP
    cpu=32,                            # Sufficient CPUs
    memory=128 * 1024,                 # 128 GB RAM
    timeout=14400,                     # 4 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/outputs": volume,
        "/vstar_cache": cache_volume
    },
)
def train_ddp(code_tar: bytes, config_yaml: str):
    """Run DDP training on 4× H100 GPUs"""
    import subprocess
    import sys
    import tarfile
    import io

    print("=" * 80, flush=True)
    print(" DDP: PREPARING MULTI-GPU TRAINING FOR MODAL", flush=True)
    print("=" * 80, flush=True)
    print(" Technique: DistributedDataParallel (DDP)", flush=True)
    print(" GPUs: 4× H100", flush=True)
    print(" Comparison: Benchmarking parallel training", flush=True)
    print("=" * 80, flush=True)
    print()

    # Setup workspace
    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Extract code
    print("Extracting code...", flush=True)
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()
    print("Code extracted", flush=True)
    print()

    # Write config
    print("Writing DDP config...", flush=True)
    os.makedirs("configs", exist_ok=True)
    with open("configs/ddp_config.yaml", "w") as f:
        f.write(config_yaml)
    print("Config written", flush=True)

    # Display config details
    import yaml
    with open("configs/ddp_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("   - Batch size per GPU:", cfg['apo']['batch_size'])
    print("   - Num GPUs: 4")
    print("   - Gradient accumulation:", cfg['apo']['gradient_accumulation_steps'])
    effective_batch = cfg['apo']['batch_size'] * 4 * cfg['apo']['gradient_accumulation_steps']
    print(f"   - Effective batch size: {cfg['apo']['batch_size']} × 4 × {cfg['apo']['gradient_accumulation_steps']} = {effective_batch}")
    print()

    # Install package
    print("Installing tinyzero package...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        capture_output=True
    )
    print("Package installed", flush=True)
    print()

    # Check GPU availability
    print("Checking GPU availability...", flush=True)
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")
    print()

    # Setup distributed environment
    print("Setting up distributed environment...", flush=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '4'
    print("Environment configured", flush=True)
    print()

    # Launch DDP training with torchrun
    print("=" * 80, flush=True)
    print(" STARTING DDP TRAINING", flush=True)
    print("=" * 80, flush=True)
    print()

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--master_addr=localhost",
        "--master_port=29500",
        "-m",
        "tinyzero.train_ddp",
        "--config", "configs/ddp_config.yaml",
        "--output_dir", "/outputs"
    ]

    print(f"Command: {' '.join(cmd)}", flush=True)
    print("=" * 80, flush=True)
    print()
    sys.stdout.flush()

    # Run with real-time output
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise Exception(f"DDP training failed with code {result.returncode}")

    print()
    print("=" * 80, flush=True)
    print(" TRAINING COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print()

    # Commit volumes
    print("Committing volumes...", flush=True)
    volume.commit()
    cache_volume.commit()
    print("Volumes committed", flush=True)

    return {"status": "success", "output_dir": "/outputs"}


@app.local_entrypoint()
def main():
    """Local entry point"""
    import tarfile
    import io
    from pathlib import Path

    print()
    print("=" * 80)
    print(" DDP: PREPARING MULTI-GPU TRAINING FOR MODAL")
    print("=" * 80)
    print(" Technique: DistributedDataParallel (DDP)")
    print(" GPUs: 4× H100")
    print(" Comparison: Benchmarking parallel training")
    print("=" * 80)
    print()

    # Check we're in right directory
    if not Path("tinyzero").exists():
        print("Error: Error: Run from project root directory!")
        return

    # Create tarball of code
    print("Packaging code...")
    print("   - Adding tinyzero/")
    print("   - Adding setup.py")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add('tinyzero', arcname='tinyzero')
        tar.add('setup.py', arcname='setup.py')

    code_tar = tar_buffer.getvalue()
    print(f"Packaged {len(code_tar):,} bytes ({len(code_tar) / 1024 / 1024:.1f} MB)")
    print()

    # Load config
    print("Loading DDP config...")
    config_path = Path("configs/ddp_config.yaml")
    if not config_path.exists():
        print(f"Error: Error: Config not found at {config_path}")
        return

    with open(config_path) as f:
        config_yaml = f.read()

    import yaml
    cfg = yaml.safe_load(config_yaml)

    print("Config loaded")
    print("   - Batch size per GPU:", cfg['apo']['batch_size'])
    print("   - Num GPUs: 4")
    print("   - Gradient accumulation:", cfg['apo']['gradient_accumulation_steps'])
    effective_batch = cfg['apo']['batch_size'] * 4 * cfg['apo']['gradient_accumulation_steps']
    print(f"   - Effective batch size: {cfg['apo']['batch_size']} × 4 × {cfg['apo']['gradient_accumulation_steps']} = {effective_batch}")
    print()

    # Launch on Modal
    print("=" * 80)
    print(" READY TO LAUNCH ON MODAL")
    print("=" * 80)
    print(" Configuration:")
    print("   - GPUs: 4 × H100")
    print("   - Backend: NCCL")
    print("   - CPUs: 32")
    print("   - RAM: 128 GB")
    print("   - Timeout: 4 hours")
    print("=" * 80)
    print()
    print("Launching DDP training on Modal...")
    print("   (This will take several minutes to provision GPUs)")
    print()

    result = train_ddp.remote(code_tar, config_yaml)

    print()
    print("=" * 80)
    print(" DONE!")
    print("=" * 80)
    print(f"Result: {result}")
    print()
    print("To download results:")
    print("  modal volume get tinyzero-outputs /outputs ./results")
