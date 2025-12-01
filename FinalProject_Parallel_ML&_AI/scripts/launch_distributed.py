"""
Launch script for distributed multi-GPU training.

Usage:
    python scripts/launch_distributed.py --num-gpus 4 --config configs/ddp_config.yaml

This script uses torch.distributed.launch to spawn multiple processes,
one per GPU, for distributed training with PyTorch DDP.
"""
import argparse
import sys
import os
import subprocess
import torch
from pathlib import Path


def check_gpu_availability(num_gpus: int) -> bool:
    """Check if requested number of GPUs are available."""
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Cannot run distributed training.")
        return False

    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} GPU(s)")

    if num_gpus > available_gpus:
        print(f"Error: Requested {num_gpus} GPUs but only {available_gpus} available")
        return False

    return True


def launch_distributed(num_gpus: int, config_path: str, extra_args: list = None):
    """
    Launch distributed training using torchrun (new) or torch.distributed.launch (legacy).

    Args:
        num_gpus: Number of GPUs to use
        config_path: Path to config file
        extra_args: Additional arguments to pass to training script
    """
    # Get project root
    project_root = Path(__file__).parent.parent

    # Training script path
    train_script = project_root / "tinyzero" / "train_distributed.py"

    if not train_script.exists():
        print(f"Error: Training script not found: {train_script}")
        print("   Creating placeholder...")
        # Will create this next
        return False

    # Build command
    # Try torchrun first (newer, recommended)
    try:
        torchrun_cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            str(train_script),
            "--config", config_path,
        ]

        if extra_args:
            torchrun_cmd.extend(extra_args)

        print("\n" + "="*80)
        print("LAUNCHING DISTRIBUTED TRAINING")
        print("="*80)
        print(f"Command: {' '.join(torchrun_cmd)}")
        print(f"GPUs: {num_gpus}")
        print(f"Config: {config_path}")
        print("="*80 + "\n")

        # Launch
        result = subprocess.run(torchrun_cmd, cwd=project_root)
        return result.returncode == 0

    except FileNotFoundError:
        # Fall back to legacy torch.distributed.launch
        print("torchrun not found, trying legacy torch.distributed.launch...")

        launch_cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            str(train_script),
            "--config", config_path,
        ]

        if extra_args:
            launch_cmd.extend(extra_args)

        print("\n" + "="*80)
        print("LAUNCHING DISTRIBUTED TRAINING (Legacy)")
        print("="*80)
        print(f"Command: {' '.join(launch_cmd)}")
        print(f"GPUs: {num_gpus}")
        print(f"Config: {config_path}")
        print("="*80 + "\n")

        result = subprocess.run(launch_cmd, cwd=project_root)
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Launch distributed multi-GPU training')

    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (default: all available)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ddp_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check GPU availability, do not launch training'
    )

    args, extra_args = parser.parse_known_args()

    # Determine number of GPUs
    if args.num_gpus is None:
        if torch.cuda.is_available():
            args.num_gpus = torch.cuda.device_count()
            print(f"Using all available GPUs: {args.num_gpus}")
        else:
            print("No GPUs available")
            return 1

    # Check GPU availability
    if not check_gpu_availability(args.num_gpus):
        return 1

    if args.check_only:
        print("GPU check passed")
        return 0

    # Launch distributed training
    success = launch_distributed(args.num_gpus, args.config, extra_args)

    if success:
        print("\n" + "="*80)
        print("DISTRIBUTED TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("Error: DISTRIBUTED TRAINING FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
