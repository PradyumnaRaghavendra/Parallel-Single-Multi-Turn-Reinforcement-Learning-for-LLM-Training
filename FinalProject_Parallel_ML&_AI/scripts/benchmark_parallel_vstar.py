"""
Benchmark Script for Phase 1: Parallel V* Computation

This script compares:
1. Sequential V* computation (baseline)
2. Parallel V* computation (Phase 1 optimization)

Expected speedup: 5-6x for V* computation
"""
import sys
import time
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyzero.models import ReferenceModel
from tinyzero.vstar_cache import VStarCache
from tinyzero.data import CurriculumMathDataset
from tinyzero.parallel_vstar import ParallelVStarComputer, compute_vstar_parallel
import torch


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def benchmark_sequential_vstar(
    prompts,
    problems,
    ref_model,
    config,
    num_samples=5
):
    """
    Benchmark sequential V* computation (original method)
    """
    print("\n" + "="*80)
    print("SEQUENTIAL V* COMPUTATION (Baseline)")
    print("="*80)

    from tinyzero.rewards import compute_reward

    start_time = time.time()
    V_star_values = []

    for i, prompt in enumerate(prompts):
        # Generate samples
        samples = ref_model.generate(
            [prompt],
            num_samples=num_samples,
            min_new_tokens=30,
            temperature=1.0,
            max_length=config['model']['max_length']
        )[0]

        # Compute rewards
        problem = problems[i] if problems else {'prompt': prompt, 'task': 'math'}
        rewards = [compute_reward(s, problem, require_cot=False) for s in samples]
        rewards = np.array(rewards, dtype=np.float32)

        # Calculate V*
        beta = config['apo']['beta']
        if rewards.size == 0:
            V_star = 0.0
        else:
            if beta > 0:
                max_r = rewards.max()
                exp_terms = np.exp((rewards - max_r) / beta)
                V_star = float(max_r + beta * np.log(np.mean(exp_terms)))
            else:
                V_star = float(rewards.max())

        V_star_values.append(V_star)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(prompts)} prompts...")

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\nSequential V* completed")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Time per prompt: {elapsed/len(prompts):.3f} seconds")
    print(f"  Throughput: {len(prompts)/elapsed:.2f} prompts/second")

    return np.array(V_star_values), elapsed


def benchmark_parallel_vstar(
    prompts,
    problems,
    config,
    num_samples=5,
    num_workers=8
):
    """
    Benchmark parallel V* computation (Phase 1 optimization)
    """
    print("\n" + "="*80)
    print(f"PARALLEL V* COMPUTATION (Phase 1 - {num_workers} workers)")
    print("="*80)

    ref_model_name = config['model']['ref_model']

    start_time = time.time()

    # Create parallel computer
    computer = ParallelVStarComputer(
        ref_model_name=ref_model_name,
        config=config,
        num_workers=num_workers,
        device_per_worker='cpu'
    )

    # Compute V* in parallel
    V_star_values, all_rewards = computer.compute_batch(
        prompts=prompts,
        num_samples=num_samples,
        problems=problems
    )

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\nParallel V* completed")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Time per prompt: {elapsed/len(prompts):.3f} seconds")
    print(f"  Throughput: {len(prompts)/elapsed:.2f} prompts/second")

    return V_star_values, elapsed


def run_benchmark(
    config_path: str,
    num_prompts: int = 20,
    num_samples: int = 5,
    worker_counts: list = [1, 2, 4, 8]
):
    """
    Run comprehensive benchmark comparing sequential and parallel V*
    """
    print("\n" + "="*80)
    print("PHASE 1 BENCHMARK: Parallel V* Computation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Number of prompts: {num_prompts}")
    print(f"  Samples per prompt: {num_samples}")
    print(f"  Worker counts to test: {worker_counts}")

    # Load config
    config = load_config(config_path)

    # Create dataset to get test prompts
    print("\n" + "-"*80)
    print("Loading dataset and reference model...")
    print("-"*80)

    dataset = CurriculumMathDataset(
        num_samples=num_prompts,
        tasks=config['data']['tasks']
    )

    # Get test data
    problems = [dataset[i] for i in range(min(num_prompts, len(dataset)))]
    prompts = [p['prompt'] for p in problems]

    print(f"Loaded {len(prompts)} test prompts")
    print(f"\nExample prompts:")
    for i in range(min(3, len(prompts))):
        print(f"  {i+1}. {prompts[i]}")

    # Load reference model for sequential benchmark
    print(f"\nLoading reference model: {config['model']['ref_model']}")
    ref_model = ReferenceModel(
        config['model']['ref_model'],
        device='cpu',  # Use CPU for fair comparison
        max_input_len=config['model']['max_length']
    )
    print("Reference model loaded")

    # ========== SEQUENTIAL BENCHMARK ==========
    seq_vstar, seq_time = benchmark_sequential_vstar(
        prompts=prompts,
        problems=problems,
        ref_model=ref_model,
        config=config,
        num_samples=num_samples
    )

    # ========== PARALLEL BENCHMARKS ==========
    results = {
        'sequential': {
            'time': seq_time,
            'vstar': seq_vstar,
            'workers': 1
        }
    }

    for num_workers in worker_counts:
        par_vstar, par_time = benchmark_parallel_vstar(
            prompts=prompts,
            problems=problems,
            config=config,
            num_samples=num_samples,
            num_workers=num_workers
        )

        results[f'parallel_{num_workers}w'] = {
            'time': par_time,
            'vstar': par_vstar,
            'workers': num_workers
        }

    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    print(f"\nConfiguration: {num_prompts} prompts, {num_samples} samples each")
    print("\n" + "-"*80)
    print(f"{'Method':<25} {'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Throughput (p/s)':<20}")
    print("-"*80)

    baseline_time = seq_time

    for method_name, result in results.items():
        time_taken = result['time']
        workers = result['workers']
        speedup = baseline_time / time_taken
        throughput = len(prompts) / time_taken

        print(f"{method_name:<25} {workers:<10} {time_taken:<12.2f} {speedup:<10.2f}x {throughput:<20.2f}")

    # ========== V* CORRECTNESS CHECK ==========
    print("\n" + "-"*80)
    print("V* Correctness Verification")
    print("-"*80)

    # Compare parallel results with sequential baseline
    for method_name, result in results.items():
        if method_name == 'sequential':
            continue

        par_vstar = result['vstar']
        diff = np.abs(par_vstar - seq_vstar)
        max_diff = diff.max()
        mean_diff = diff.mean()

        print(f"\n{method_name}:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print(f"  Results match sequential baseline")
        else:
            print(f"  Warning: Results differ from baseline (expected due to randomness)")

    # ========== RECOMMENDATIONS ==========
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_speedup = 0
    best_workers = 1

    for method_name, result in results.items():
        if method_name == 'sequential':
            continue

        speedup = baseline_time / result['time']
        if speedup > best_speedup:
            best_speedup = speedup
            best_workers = result['workers']

    print(f"\nBest configuration: {best_workers} workers")
    print(f"  Speedup: {best_speedup:.2f}x")
    print(f"  Time saved: {baseline_time - (baseline_time/best_speedup):.2f} seconds")

    if best_speedup >= 4.0:
        print(f"\nExcellent speedup achieved! Ready for production use.")
    elif best_speedup >= 2.0:
        print(f"\nGood speedup achieved. Consider tuning worker count.")
    else:
        print(f"\nWarning: Speedup below expectations. Check system resources.")

    # ========== SAVE RESULTS ==========
    results_file = Path(__file__).parent.parent / 'benchmark_results_phase1.txt'

    with open(results_file, 'w') as f:
        f.write("PHASE 1 BENCHMARK RESULTS: Parallel V* Computation\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration: {num_prompts} prompts, {num_samples} samples each\n\n")
        f.write(f"{'Method':<25} {'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Throughput (p/s)':<20}\n")
        f.write("-"*80 + "\n")

        for method_name, result in results.items():
            time_taken = result['time']
            workers = result['workers']
            speedup = baseline_time / time_taken
            throughput = len(prompts) / time_taken
            f.write(f"{method_name:<25} {workers:<10} {time_taken:<12.2f} {speedup:<10.2f}x {throughput:<20.2f}\n")

        f.write(f"\nBest configuration: {best_workers} workers ({best_speedup:.2f}x speedup)\n")

    print(f"\nResults saved to {results_file}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark parallel V* computation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/parallel_vstar_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=20,
        help='Number of prompts to test (default: 20)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Samples per prompt (default: 5)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='Worker counts to test (default: 1 2 4 8)'
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        config_path=args.config,
        num_prompts=args.num_prompts,
        num_samples=args.num_samples,
        worker_counts=args.workers
    )

    print("\nBenchmark completed successfully!")
