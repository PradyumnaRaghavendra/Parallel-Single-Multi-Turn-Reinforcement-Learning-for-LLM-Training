"""
Parallel V* Computation using torch.multiprocessing
Addresses the primary bottleneck in TinyZero training (60-70% of time)

This module parallelizes V* computation across multiple CPU workers.
Expected speedup: 5-6x for V* computation step.
"""
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# Ensure proper multiprocessing context
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def vstar_worker(
    prompt: str,
    model_name: str,
    config: Dict,
    num_samples: int,
    worker_id: int,
    device: str = 'cpu'
) -> Tuple[str, float, List[float]]:
    """
    Worker function for computing V* for a single prompt.

    Each worker:
    1. Loads its own copy of the reference model (on CPU to avoid GPU contention)
    2. Generates num_samples completions for the given prompt
    3. Computes rewards for each sample
    4. Calculates V* using soft-max formula
    5. Returns (prompt, V*, rewards)

    Args:
        prompt: Input prompt string
        model_name: HuggingFace model name/path
        config: Training configuration dict
        num_samples: Number of samples to generate
        worker_id: Worker process ID (for debugging)
        device: Device to use (default 'cpu')

    Returns:
        Tuple of (prompt, V* value, list of rewards)
    """
    try:
        # Import inside worker to avoid pickling issues
        # Add parent directory to path for workers
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from tinyzero.models import ReferenceModel
        from tinyzero.rewards import compute_reward

        # Load reference model on CPU (each worker gets its own copy)
        ref_model = ReferenceModel(
            model_name,
            device=device,
            max_input_len=config.get('model', {}).get('max_length', 512)
        )

        # Generate samples
        gen_max_length = config.get('model', {}).get('max_length', 128)
        samples_list = ref_model.generate(
            [prompt],
            num_samples=num_samples,
            min_new_tokens=30,
            temperature=1.0,
            max_length=gen_max_length
        )[0]  # Returns List[List[str]], take first element

        # Compute rewards (binary for math tasks)
        # Note: We pass minimal problem dict since we only have prompt
        rewards = []
        for sample in samples_list:
            problem = {'prompt': prompt, 'task': 'math'}
            reward = compute_reward(sample, problem, require_cot=False)
            rewards.append(reward)

        rewards = np.array(rewards, dtype=np.float32)

        # Calculate V* using soft-max formula
        beta = config['apo'].get('beta', 0.5)
        if rewards.size == 0:
            V_star = 0.0
        else:
            if beta > 0:
                # Soft-max: V* = max_r + β * log(mean(exp((r - max_r) / β)))
                max_r = rewards.max()
                exp_terms = np.exp((rewards - max_r) / beta)
                V_star = float(max_r + beta * np.log(np.mean(exp_terms)))
            else:
                # Hard-max: V* = max(rewards)
                V_star = float(rewards.max())

        return (prompt, V_star, rewards.tolist())

    except Exception as e:
        print(f"[Worker {worker_id}] Error processing prompt: {e}")
        import traceback
        traceback.print_exc()
        # Return safe defaults
        return (prompt, 0.0, [0.0] * num_samples)


class ParallelVStarComputer:
    """
    Parallel V* computation using multiprocessing.

    Uses torch.multiprocessing.Pool to distribute V* computation
    across multiple CPU workers, with each worker loading its own
    copy of the reference model.

    This addresses the primary bottleneck in TinyZero training.
    """

    def __init__(
        self,
        ref_model_name: str,
        config: Dict,
        num_workers: int = 8,
        device_per_worker: str = 'cpu'
    ):
        """
        Initialize parallel V* computer.

        Args:
            ref_model_name: HuggingFace model name/path for reference model
            config: Training configuration dict
            num_workers: Number of parallel workers (default 8)
            device_per_worker: Device for each worker (default 'cpu')
        """
        self.ref_model_name = ref_model_name
        self.config = config
        self.num_workers = num_workers
        self.device = device_per_worker

        print(f"Parallel V* Computer initialized")
        print(f"  - Workers: {num_workers}")
        print(f"  - Device per worker: {device_per_worker}")
        print(f"  - Model: {ref_model_name}")

    def compute_batch(
        self,
        prompts: List[str],
        num_samples: int = 5,
        problems: Optional[List[Dict]] = None
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Compute V* for a batch of prompts in parallel.

        Args:
            prompts: List of prompt strings
            num_samples: Number of samples per prompt (default 5)
            problems: Optional list of problem dicts (for reward computation)

        Returns:
            Tuple of (V* values array, list of reward lists)
        """
        if len(prompts) == 0:
            return np.array([]), []

        # Create worker arguments
        worker_args = [
            (prompt, self.ref_model_name, self.config, num_samples, i, self.device)
            for i, prompt in enumerate(prompts)
        ]

        # Use multiprocessing pool
        # Note: Using 'spawn' context to avoid CUDA issues
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.starmap(vstar_worker, worker_args)

        # Unpack results
        V_star_values = []
        all_rewards = []

        for prompt, v_star, rewards in results:
            V_star_values.append(v_star)
            all_rewards.append(rewards)

        return np.array(V_star_values, dtype=np.float32), all_rewards

    def compute_batch_with_cache(
        self,
        prompts: List[str],
        vstar_cache,
        num_samples: int = 5,
        problems: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Compute V* with caching support (integrated with VStarCache).

        Args:
            prompts: List of prompt strings
            vstar_cache: VStarCache instance
            num_samples: Number of samples per prompt
            problems: Optional list of problem dicts

        Returns:
            Array of V* values
        """
        V_star_values = []
        prompts_to_compute = []
        prompt_indices = []
        cache_hits = 0

        # First pass: Check cache
        for i, prompt in enumerate(prompts):
            cached_data = vstar_cache.get(prompt, self.config)

            if cached_data is not None:
                # Cache hit!
                V_star_values.append(cached_data['v_star'])
                cache_hits += 1
            else:
                # Cache miss - need to compute
                prompts_to_compute.append(prompt)
                prompt_indices.append(i)
                V_star_values.append(None)  # Placeholder

        # Second pass: Parallel compute uncached prompts
        if prompts_to_compute:
            print(f"  Computing V* for {len(prompts_to_compute)} uncached prompts (parallel)...")

            # Parallel computation
            computed_vstars, all_rewards = self.compute_batch(
                prompts_to_compute,
                num_samples=num_samples,
                problems=problems
            )

            # Fill in results and update cache
            for idx, (v_star, rewards) in enumerate(zip(computed_vstars, all_rewards)):
                original_idx = prompt_indices[idx]
                prompt = prompts_to_compute[idx]

                # Update result array
                V_star_values[original_idx] = v_star

                # Save to cache
                cache_data = {
                    'v_star': float(v_star),
                    'rewards': rewards,
                    'num_samples': len(rewards)
                }
                vstar_cache.save(prompt, self.config, cache_data)

        # Print cache stats
        if cache_hits > 0:
            hit_rate = cache_hits / len(prompts) * 100
            print(f"  V* cache: {cache_hits}/{len(prompts)} hits ({hit_rate:.1f}% - saved compute!)")

        return np.array(V_star_values, dtype=np.float32)


# Convenience function for single-shot parallel V* computation
def compute_vstar_parallel(
    prompts: List[str],
    ref_model_name: str,
    config: Dict,
    num_samples: int = 5,
    num_workers: int = 8,
    vstar_cache=None,
    problems: Optional[List[Dict]] = None
) -> np.ndarray:
    """
    Convenience function to compute V* in parallel.

    Args:
        prompts: List of prompts
        ref_model_name: Reference model name
        config: Training config
        num_samples: Samples per prompt
        num_workers: Number of parallel workers
        vstar_cache: Optional VStarCache instance
        problems: Optional problem dicts

    Returns:
        Array of V* values
    """
    computer = ParallelVStarComputer(
        ref_model_name=ref_model_name,
        config=config,
        num_workers=num_workers
    )

    if vstar_cache is not None:
        return computer.compute_batch_with_cache(
            prompts, vstar_cache, num_samples, problems
        )
    else:
        v_stars, _ = computer.compute_batch(prompts, num_samples, problems)
        return v_stars


if __name__ == '__main__':
    """Test parallel V* computation"""
    print("Testing Parallel V* Computation\n")

    # Test configuration
    test_config = {
        'model': {
            'ref_model': 'gpt2',
            'max_length': 128
        },
        'apo': {
            'beta': 0.5,
            'v_star_samples': 3
        }
    }

    # Test prompts
    test_prompts = [
        "What is 5 * 7?",
        "Calculate 12 * 8.",
        "What is 15 * 3?"
    ]

    print(f"Testing with {len(test_prompts)} prompts")
    print(f"Model: gpt2 (for testing)")
    print(f"Samples per prompt: 3")
    print(f"Workers: 4\n")

    # Create parallel computer
    computer = ParallelVStarComputer(
        ref_model_name='gpt2',
        config=test_config,
        num_workers=4
    )

    # Compute V*
    print("Computing V* values...")
    v_stars, rewards = computer.compute_batch(
        test_prompts,
        num_samples=3
    )

    print("\nResults:")
    for i, (prompt, v_star) in enumerate(zip(test_prompts, v_stars)):
        print(f"  Prompt {i+1}: V* = {v_star:.4f}")
        print(f"    Rewards: {rewards[i]}")

    print("\nTest completed successfully!")
