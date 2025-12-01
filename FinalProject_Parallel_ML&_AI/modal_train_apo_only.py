"""
Multi-Turn APO Training for WebShop RAGEN (NO BC!)

Skips BC entirely - trains with pure RL from base model.
Goal: Demonstrate RL works (even 1% improvement is success!)
"""

import modal
import sys
from pathlib import Path

# Modal setup
app = modal.App("webshop-apo-only")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.37.0",
        "accelerate==0.27.0",
        "numpy==1.24.3",
        "gym==0.26.2",
    )
)

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_apo_only(code_tar: bytes):
    """Train with pure Multi-Turn APO (no BC warm-start)"""
    import torch
    import tarfile
    import io
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Extract code
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall("/root")
    sys.path.insert(0, "/root/FinalProject_Parallel_ML&_AI")

    from ragen.environments.simple_webshop_mock import SimpleMockWebShop
    from tinyzero.multiturn_apo_trainer import MultiTurnAPOTrainer
    from tinyzero.models import PolicyModel

    print("="*80)
    print("PURE MULTI-TURN APO TRAINING (NO BC!)")
    print("="*80)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_props.name}")
        print(f"Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    print()

    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading model: {model_name}")

    # Load as PolicyModel for APO compatibility
    policy_model = PolicyModel(model_name)
    tokenizer = policy_model.tokenizer
    model = policy_model.model

    print(f"Model loaded: {model.num_parameters() / 1e9:.2f}B parameters")
    print()

    # Setup
    env = SimpleMockWebShop()

    print(f"Environment: {len(env.products)} products")
    print()

    # Training hyperparameters
    learning_rate = 1e-5
    apo_episodes = 50  # More episodes since no BC warm-start

    # =========================================================================
    # EVALUATION
    # =========================================================================
    def evaluate(model, tokenizer, env, num_episodes=20):
        """Evaluate model performance"""
        model.eval()
        successes = 0
        total_reward = 0.0

        with torch.no_grad():
            for _ in range(num_episodes):
                env.reset()
                task = env.instruction
                done = False
                episode_reward = 0.0

                while not done:
                    obs = env.get_obs()
                    prompt = f"Task: {task}\\nObservation: {obs}\\nAction:"

                    # Generate action
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    action = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                    # Step environment
                    result = env.step(action)
                    if len(result) == 4:
                        _, reward, done, _ = result
                    else:
                        _, reward, done = result[0], result[1], result[2]

                    episode_reward += reward

                if episode_reward > 1.5:
                    successes += 1
                total_reward += episode_reward

        model.train()
        success_rate = successes / num_episodes * 100
        avg_reward = total_reward / num_episodes
        return success_rate, avg_reward

    # =========================================================================
    # TRAINING PIPELINE
    # =========================================================================
    print("-"*80)
    print("Base Model Evaluation")
    print("-"*80)
    success_base, reward_base = evaluate(model, tokenizer, env)
    print(f"Base model - Success: {success_base:.1f}%, Avg reward: {reward_base:.3f}")
    print()

    print("-"*80)
    print(f"Multi-Turn APO Training ({apo_episodes} episodes)")
    print("-"*80)

    # Create reference model (frozen copy of base model)
    print("Creating reference model (frozen copy of base model)...")
    ref_model = PolicyModel(model_name)
    ref_model.model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad = False
    print()

    # APO config
    apo_config = {
        'learning_rate': learning_rate,
        'apo': {
            'beta': 0.1,
            'gamma': 1.0,
            'gae_lambda': 1.0,
        },
        'model': {
            'max_length': 128,
            'sft_max_length': 256,
        },
        'sampling': {
            'temperature': 0.7,
            'top_p': 0.9,
        },
    }

    # Initialize Multi-Turn APO trainer
    apo_trainer = MultiTurnAPOTrainer(
        policy_model=policy_model,
        reference_model=ref_model,
        config=apo_config
    )

    # Training loop
    successes = 0
    total_reward = 0.0

    for episode in range(apo_episodes):
        # Train on 1 episode at a time
        loss, stats = apo_trainer.train_step(env, num_episodes=1)

        episode_reward = stats.get('avg_reward', 0.0)
        if episode_reward > 1.5:
            successes += 1
        total_reward += episode_reward

        if (episode + 1) % 5 == 0:
            success_rate = successes / (episode + 1) * 100
            avg_reward = total_reward / (episode + 1)
            print(f"  APO Episode {episode+1}/{apo_episodes} - Success: {success_rate:.1f}%, Avg reward: {avg_reward:.3f}, Loss: {loss:.4f}")

    print("APO training complete!")
    print()

    success_apo, reward_apo = evaluate(model, tokenizer, env)
    print(f"After APO - Success: {success_apo:.1f}%, Avg reward: {reward_apo:.3f}")
    print()

    # Summary
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Base:      {success_base:.1f}% success, {reward_base:.3f} reward")
    print(f"After APO: {success_apo:.1f}% success, {reward_apo:.3f} reward")
    print(f"APO Improvement: {success_apo - success_base:+.1f}%")
    print()

    if success_apo > success_base:
        improvement_pct = ((success_apo - success_base) / max(success_base, 1)) * 100
        print(f"SUCCESS! APO improved by {improvement_pct:.1f}% relative improvement!")
        print(f"Absolute improvement: {success_apo - success_base:+.1f} percentage points")
    else:
        print(f"APO did not improve over base model")
        print(f"Change: {success_apo - success_base:+.1f} percentage points")

    return {
        'base_success': success_base,
        'apo_success': success_apo,
        'improvement': success_apo - success_base,
        'base_reward': reward_base,
        'apo_reward': reward_apo,
    }


@app.local_entrypoint()
def main():
    """Package code and run training"""
    import tarfile
    import io
    import json

    print("Starting pure Multi-Turn APO training on Modal (H100 GPU)...")
    print("Goal: Demonstrate RL works (even 1% improvement is success!)")
    print()

    # Package code
    code_dir = Path(__file__).parent
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        for pattern in ['ragen', 'tinyzero']:
            for item in (code_dir / pattern).rglob('*.py'):
                arcname = str(item.relative_to(code_dir.parent))
                tar.add(item, arcname=arcname)

    code_tar = tar_buffer.getvalue()

    # Run training
    summary = train_apo_only.remote(code_tar)

    print()
    print("Results summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
