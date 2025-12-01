import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

from tinyzero.rewards import compute_reward
from ragen.uncertainty_filtering import UncertaintyFilter


class ValueNetwork(nn.Module):
    """Critic network for estimating state values in PPO."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, seq_len]
        """
        return self.value_head(hidden_states).squeeze(-1)


class PPOTrainer:
    """
    PPO Trainer with Trained Critic - Following RAGEN Paper

    Key features:
    - Trained value network (critic) for advantage estimation
    - GAE (Generalized Advantage Estimation) with γ=1.0, λ=1.0
    - Clipped surrogate objective
    - Entropy bonus for exploration
    """

    def __init__(self, policy_model, reference_model, config: Dict):
        self.policy = policy_model
        self.ref_model = reference_model
        self.config = config

        # PPO hyperparameters (RAGEN paper settings)
        ppo_cfg = config.get('ppo', {})
        self.learning_rate = ppo_cfg.get('learning_rate', 1e-5)
        self.clip_epsilon = ppo_cfg.get('clip_epsilon', 0.2)
        self.value_loss_coef = ppo_cfg.get('value_loss_coef', 0.5)
        self.entropy_coef = ppo_cfg.get('entropy_coef', 0.001)
        self.max_grad_norm = ppo_cfg.get('max_grad_norm', 1.0)

        # GAE parameters (RAGEN paper: γ=1.0, λ=1.0)
        self.gamma = ppo_cfg.get('gamma', 1.0)
        self.gae_lambda = ppo_cfg.get('gae_lambda', 1.0)

        # Generation parameters
        model_cfg = config.get('model', {})
        self.gen_max_length = model_cfg.get('max_length', 128)
        self.sft_max_length = min(
            model_cfg.get('sft_max_length', 256),
            getattr(self.policy.tokenizer, "model_max_length", 4096)
        )

        # Sampling controls
        samp = config.get('sampling', {})
        self.temperature = samp.get('temperature', 0.8)
        self.top_p = samp.get('top_p', 0.9)
        self.top_k = samp.get('top_k', 0)

        # Initialize value network (critic)
        hidden_size = self.policy.model.config.hidden_size
        self.value_net = ValueNetwork(hidden_size).to(self.policy.model.device)

        # Optimizer for both policy and value network
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.learning_rate},
            {'params': self.value_net.parameters(), 'lr': self.learning_rate * 3}  # Higher LR for critic
        ])

        # Initialize trajectory filtering (RAGEN paper: uncertainty-based filtering)
        unc_cfg = config.get('uncertainty_filtering', {})
        self.uncertainty_filter = UncertaintyFilter(
            keep_percent=unc_cfg.get('keep_percent', 0.5)
        )
        self.filter_enabled = unc_cfg.get('enabled', False)
        self.filter_start_step = unc_cfg.get('start_step', 20)

        self.step = 0
        print("PPO Trainer initialized with trained critic")
        print(f"  - Clip ε: {self.clip_epsilon}")
        print(f"  - GAE: γ={self.gamma}, λ={self.gae_lambda}")
        print(f"  - Entropy coef: {self.entropy_coef}")
        if self.filter_enabled:
            print(f"  - Trajectory filtering: ENABLED (start step {self.filter_start_step}, keep {unc_cfg.get('keep_percent', 0.5)*100}%)")

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: [B, T] - rewards at each timestep
            values: [B, T] - value estimates at each timestep
            masks: [B, T] - mask for valid tokens (1 = valid, 0 = padding)

        Returns:
            advantages: [B, T] - GAE advantages
            returns: [B, T] - discounted returns
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[:, t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + self.gamma * next_value * masks[:, t] - values[:, t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * masks[:, t] * gae
            advantages[:, t] = gae

        # Returns = Advantages + Values
        returns = advantages + values

        return advantages, returns

    def _build_concat_with_labels(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, pad_id: int):
        """Construct input_ids and labels with prompt masking."""
        device = prompt_ids.device
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()
        labels[:] = -100  # Mask all initially

        prompt_lens = (prompt_ids != pad_id).sum(dim=1)
        B, T = labels.size()
        for i in range(B):
            start = int(prompt_lens[i].item())
            if start < T:
                labels[i, start:] = input_ids[i, start:]

        return input_ids, attention_mask, labels

    def train_step(self, batch: List[Dict]) -> tuple:
        """PPO training step with trained critic."""
        device = self.policy.model.device

        # Ensure model is in training mode for gradient updates
        self.policy.model.train()

        try:
            # Apply trajectory filtering if enabled (RAGEN paper: filter high-variance prompts)
            if self.filter_enabled and self.uncertainty_filter.should_filter(self.step):
                # Filter batch to keep only high-uncertainty trajectories
                original_size = len(batch)
                batch, _, filter_stats = self.uncertainty_filter.filter_batch(batch, batch)
                if len(batch) == 0:
                    print("Warning: Filtering removed all trajectories, skipping step")
                    return 0.0, {
                        'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                        'entropy': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                        'avg_value': 0.0
                    }
                if self.step % 5 == 0:  # Log filtering occasionally
                    print(f"  📊 Filtered: kept {filter_stats['kept']}/{original_size} high-uncertainty trajectories")

            prompts = [item['prompt'] for item in batch]

            # Check if we have pre-generated rollouts with rewards
            if 'generated_text' in batch[0] and 'reward' in batch[0]:
                # Use pre-generated rollouts from multi-turn interaction
                generated_texts = [item['generated_text'] for item in batch]
                rewards = [item['reward'] for item in batch]
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            else:
                # Step 1: Generate trajectories from current policy
                self.policy.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                generated_texts = self.policy.generate(
                    prompts,
                    max_length=self.gen_max_length,
                    min_new_tokens=30,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=self.top_p,
                    top_k=self.top_k
                )

                # Step 2: Compute rewards
                rewards = [compute_reward(text, problem, require_cot=False)
                          for text, problem in zip(generated_texts, batch)]
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Step 3: Build training batch
            enc_prompts = self.policy.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.sft_max_length
            )
            enc_comps = self.policy.tokenizer(
                generated_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.sft_max_length,
                add_special_tokens=False
            )
            enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
            enc_comps = {k: v.to(device) for k, v in enc_comps.items()}

            pad_id = self.policy.tokenizer.pad_token_id or self.policy.tokenizer.eos_token_id
            input_ids, attention_mask, labels = self._build_concat_with_labels(
                enc_prompts["input_ids"], enc_comps["input_ids"], pad_id
            )

            # Truncate to max length
            input_ids = input_ids[:, :self.sft_max_length]
            attention_mask = attention_mask[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # Step 4: Get old policy logprobs (for PPO ratio)
            with torch.no_grad():
                old_outputs = self.policy.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                old_logits = old_outputs.logits
                old_logprobs = F.log_softmax(old_logits, dim=-1)

                # Get logprobs of actual tokens
                B, T, V = old_logits.shape
                shift_labels = labels[:, 1:].contiguous()
                shift_logprobs = old_logprobs[:, :-1, :].contiguous()

                # Clamp token IDs to valid vocabulary range to prevent CUDA errors
                safe_labels_old = shift_labels.clamp(0, V - 1)

                old_action_logprobs = torch.gather(
                    shift_logprobs.view(-1, V),
                    1,
                    safe_labels_old.view(-1, 1)
                ).view(B, T - 1)

                # Mask invalid tokens
                token_mask = (shift_labels != -100).float()
                old_action_logprobs = old_action_logprobs * token_mask

            # Step 5: Forward pass with new policy
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

            # Get new policy logprobs
            new_logprobs = F.log_softmax(logits, dim=-1)
            shift_logprobs = new_logprobs[:, :-1, :].contiguous()

            # Clamp token IDs to valid vocabulary range to prevent CUDA errors
            safe_labels = shift_labels.clamp(0, V - 1)

            new_action_logprobs = torch.gather(
                shift_logprobs.view(-1, V),
                1,
                safe_labels.view(-1, 1)
            ).view(B, T - 1)
            new_action_logprobs = new_action_logprobs * token_mask

            # Step 6: Compute values from critic
            values = self.value_net(hidden_states)[:, :-1]  # [B, T-1]

            # Step 7: Assign trajectory-level rewards to all tokens
            # In multi-turn RL, the reward comes at the end of trajectory
            rewards_per_token = rewards_t.unsqueeze(1).expand_as(values)  # [B, T-1]
            rewards_per_token = rewards_per_token * token_mask  # Mask padding

            # Step 8: Compute GAE advantages
            advantages, returns = self.compute_gae(
                rewards_per_token,
                values.detach(),
                token_mask
            )

            # Normalize advantages
            if token_mask.sum() > 0:
                adv_mean = (advantages * token_mask).sum() / token_mask.sum()
                adv_std = torch.sqrt(((advantages - adv_mean) ** 2 * token_mask).sum() / token_mask.sum()) + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            # Step 9: Compute PPO clipped loss
            # Importance ratio: π_new / π_old
            log_ratio = new_action_logprobs - old_action_logprobs.detach()
            ratio = torch.exp(log_ratio)

            # Clipped surrogate loss
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2) * token_mask
            policy_loss = policy_loss.sum() / token_mask.sum()

            # Step 10: Value loss (MSE between predicted and target returns)
            value_loss = F.mse_loss(values * token_mask, returns.detach() * token_mask, reduction='sum')
            value_loss = value_loss / token_mask.sum()

            # Step 11: Entropy bonus for exploration
            entropy = -(new_logprobs * torch.exp(new_logprobs)).sum(dim=-1)[:, :-1]
            entropy = (entropy * token_mask).sum() / token_mask.sum()

            # Step 12: Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # Step 13: Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Metrics
            self.step += 1
            avg_reward = float(rewards_t.mean().item())

            # Guard against division by zero when no valid tokens
            num_tokens = token_mask.sum().item()
            if num_tokens > 0:
                avg_advantage = float((advantages * token_mask).sum().item() / num_tokens)
                avg_value = float((values * token_mask).sum().item() / num_tokens)
            else:
                # No valid tokens in batch - use zeros
                avg_advantage = 0.0
                avg_value = 0.0
                print(f"Warning: Warning: No valid tokens in batch at step {self.step}")

            if self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print(
                    f"Step {self.step}: "
                    f"Loss={loss.item():.4f} (Policy={policy_loss.item():.4f}, "
                    f"Value={value_loss.item():.4f}, Entropy={entropy.item():.4f}), "
                    f"Reward={avg_reward:.3f}, "
                    f"Advantage={avg_advantage:.3f}, "
                    f"Value={avg_value:.3f}"
                )

            stats = {
                'loss': float(loss.item()),
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.item()),
                'avg_reward': avg_reward,
                'avg_advantage': avg_advantage,
                'avg_value': avg_value,
            }

            return float(loss.item()), stats

        except Exception as e:
            print(f"\n--- Error in PPO train_step ---")
            print(f"Step: {self.step}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("--------------------------------\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return 0.0, {
                'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                'entropy': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_value': 0.0
            }

    def sft_step(self, prompts: List[str], responses: List[str]) -> float:
        """Supervised fine-tuning step (behavior cloning warm-start)."""
        device = self.policy.model.device

        enc_prompts = self.policy.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.sft_max_length
        )
        enc_resps = self.policy.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.sft_max_length,
            add_special_tokens=False
        )

        enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
        enc_resps = {k: v.to(device) for k, v in enc_resps.items()}

        pad_id = self.policy.tokenizer.pad_token_id or self.policy.tokenizer.eos_token_id
        input_ids, attention_mask, labels = self._build_concat_with_labels(
            enc_prompts["input_ids"], enc_resps["input_ids"], pad_id
        )

        input_ids = input_ids[:, :self.sft_max_length]
        attention_mask = attention_mask[:, :self.sft_max_length]
        labels = labels[:, :self.sft_max_length]

        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return float(loss.item())

    def behavior_cloning_warmstart(self, num_steps: int = 30):
        """
        Behavior cloning warm-start using expert demonstrations.
        Compatible with RAGEN training loop interface.
        """
        print(f"\n🎓 BEHAVIOR CLONING WARM-START ({num_steps} steps)")
        print("=" * 60)

        # Load expert demos based on environment type
        env_type = self.config.get('environment', {}).get('type', 'maze')

        if env_type == 'maze':
            try:
                from ragen.maze_expert_demos import get_maze_expert_demos
                all_demos = get_maze_expert_demos()
                print(f"  Loaded {len(all_demos)} maze expert demos")
            except ImportError:
                print("  Warning: No expert demos found, skipping warm-start")
                return
        elif env_type == 'webshop':
            try:
                from ragen.webshop_expert_demos import format_demos_for_ppo_warmstart
                all_demos = format_demos_for_ppo_warmstart()
                print(f"  Loaded {len(all_demos)} WebShop expert demo examples")
            except ImportError:
                print("  Warning: No WebShop expert demos found, skipping warm-start")
                return
        else:
            print(f"  Warning: No expert demos for environment '{env_type}', skipping warm-start")
            return

        # Training loop
        device = self.policy.model.device
        total_loss = 0.0

        # CRITICAL: Set model to training mode
        self.policy.model.train()

        for step in range(num_steps):
            step_loss = 0.0

            for demo in all_demos:
                # Handle different demo formats
                if 'prompt' in demo and 'target_action' in demo:
                    # WebShop demo format: {prompt, target_action}
                    prompt = demo['prompt']
                    target_action = demo['target_action']

                    # Apply chat template properly
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target_action}
                    ]
                    full_text = self.policy.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )

                    # Tokenize the full conversation
                    full_enc = self.policy.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=640
                    )

                    # Tokenize just user message to find boundary
                    user_messages = [{"role": "user", "content": prompt}]
                    user_text = self.policy.tokenizer.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True  # This adds the assistant prefix
                    )
                    user_enc = self.policy.tokenizer(
                        user_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )

                    # Move to device
                    input_ids = full_enc['input_ids'].to(device)
                    attention_mask = full_enc['attention_mask'].to(device)

                    # Create labels - mask everything before assistant response
                    labels = input_ids.clone()
                    user_length = user_enc['input_ids'].shape[1]
                    labels[:, :user_length] = -100  # Mask user prompt and special tokens

                    # Mask padding tokens only (DON'T mask EOS - model needs to learn termination!)
                    if self.policy.tokenizer.pad_token_id is not None:
                        labels[labels == self.policy.tokenizer.pad_token_id] = -100

                    # Forward pass
                    outputs = self.policy.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    step_loss += loss.item()

                elif 'turns' in demo:
                    # Maze demo format: train on ALL turns (not just first!)
                    for turn in demo['turns']:
                        prompt = turn['observation']
                        target_action = turn['action']

                        # Apply chat template properly
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": target_action}
                        ]
                        full_text = self.policy.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        # Tokenize the full conversation
                        full_enc = self.policy.tokenizer(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=640
                        )

                        # Tokenize just user message to find boundary
                        user_messages = [{"role": "user", "content": prompt}]
                        user_text = self.policy.tokenizer.apply_chat_template(
                            user_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        user_enc = self.policy.tokenizer(
                            user_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )

                        # Move to device
                        input_ids = full_enc['input_ids'].to(device)
                        attention_mask = full_enc['attention_mask'].to(device)

                        # Create labels - mask everything before assistant response
                        labels = input_ids.clone()
                        user_length = user_enc['input_ids'].shape[1]
                        labels[:, :user_length] = -100

                        # Mask padding tokens only (DON'T mask EOS - model needs to learn termination!)
                        if self.policy.tokenizer.pad_token_id is not None:
                            labels[labels == self.policy.tokenizer.pad_token_id] = -100

                        # Forward pass
                        outputs = self.policy.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        loss = outputs.loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                        step_loss += loss.item()

                    # Continue to next demo after processing all turns
                    continue

                elif 'instruction' in demo:
                    # Standard demo format
                    prompt = f"Task: {demo['instruction']}\n\nWhat should you do?"
                    target_action = demo['actions'][0]  # First action is most critical
                else:
                    print(f"Warning: Skipping demo with unknown format")
                    continue

                # Apply chat template properly
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_action}
                ]
                full_text = self.policy.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Tokenize the full conversation
                full_enc = self.policy.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=640
                )

                # Tokenize just user message to find boundary
                user_messages = [{"role": "user", "content": prompt}]
                user_text = self.policy.tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                user_enc = self.policy.tokenizer(
                    user_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                # Move to device
                input_ids = full_enc['input_ids'].to(device)
                attention_mask = full_enc['attention_mask'].to(device)

                # Create labels - mask everything before assistant response
                labels = input_ids.clone()
                user_length = user_enc['input_ids'].shape[1]
                labels[:, :user_length] = -100

                # Also mask padding and EOS tokens
                if self.policy.tokenizer.pad_token_id is not None:
                    labels[labels == self.policy.tokenizer.pad_token_id] = -100
                if self.policy.tokenizer.eos_token_id is not None:
                    labels[labels == self.policy.tokenizer.eos_token_id] = -100

                # Forward pass
                outputs = self.policy.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                step_loss += loss.item()

            avg_loss = step_loss / len(all_demos)
            total_loss += avg_loss

            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps}: Loss = {avg_loss:.4f}")

        avg_total_loss = total_loss / num_steps
        print(f"\nWarm-start complete! Average loss: {avg_total_loss:.4f}")
        print("=" * 60)

        # CRITICAL: Switch model back to eval mode for generation
        self.policy.model.eval()
        print("  Model switched to eval mode for generation")
