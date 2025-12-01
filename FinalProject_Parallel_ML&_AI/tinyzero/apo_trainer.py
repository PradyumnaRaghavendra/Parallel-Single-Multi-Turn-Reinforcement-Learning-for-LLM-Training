import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional

from tinyzero.rewards import compute_reward 
# compute_reward_with_partial_credit
from tinyzero.vstar_cache import VStarCache
import math # For isnan

class APOTrainer:
    """A*PO Trainer - More Stable Advantage Weighting"""

    def __init__(self, policy_model, reference_model, config: Dict):
        self.policy = policy_model
        self.ref_model = reference_model
        self.config = config

        # APO hyperparams
        apo_cfg = config.get('apo', {})
        self.beta = apo_cfg.get('beta', 0.5)
        self.v_star_samples = apo_cfg.get('v_star_samples', 5) 
        self.learning_rate = apo_cfg.get('learning_rate', 5e-7)
        self.kl_coef = apo_cfg.get('kl_coef', 0.02)
        self.use_exp_weights = apo_cfg.get('use_exp_weights', False) # Default to Advantage Weighting
        self.adv_clip = apo_cfg.get('adv_clip', 3.0)
        self.clip_grad_norm = apo_cfg.get('clip_grad_norm', 1.0) # Grad clipping value
        self.weighting_scheme = apo_cfg.get('weighting_scheme', 'normalized_advantage') # 'exp', 'normalized_advantage', 'shifted_advantage'
        self.log_intermediate_values = apo_cfg.get('log_intermediate_values', False) # Flag for detailed logging


        # Generation / tokenization lengths
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

        # Use 8-bit Adam to save ~18 GB optimizer memory per GPU
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.policy.parameters(),
                lr=self.learning_rate
            )
            print("Using 8-bit AdamW optimizer (saves ~18 GB per GPU)")
        except ImportError:
            print("Warning: bitsandbytes not available, using standard Adam")
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.learning_rate
            )

        self.step = 0
        # V* Cache and Adaptive Sampling
        self.vstar_cache = VStarCache()
        self.adaptive_vstar = config['apo'].get('adaptive_vstar', True)
        print("V* caching enabled")
        print("Adaptive V* sampling enabled")

    @torch.no_grad()
    def compute_V_star(self, prompts: List[str], problems: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Compute V* with CACHING + ADAPTIVE SAMPLING + BATCHING.
        This is the OPTIMIZED version that saves 70%+ compute!
        """
        V_star_values = []
        cache_hits = 0
        self.ref_model.model.eval()
        
        # ===== ADAPTIVE SAMPLING =====
        # Use fewer samples as training progresses
        if self.adaptive_vstar:
            if self.step < 30:
                num_samples = 5  # Early: need accurate V*
            elif self.step < 70:
                num_samples = 3  # Mid: medium accuracy
            else:
                num_samples = 2  # Late: model is trained
        else:
            num_samples = self.v_star_samples
        
        print(f"  Computing V* for {len(prompts)} prompts ({num_samples} samples each)...")
        
        # ===== FIRST PASS: CHECK CACHE =====
        prompts_to_compute = []
        prompt_indices = []
        
        for i, prompt in enumerate(prompts):
            cached_data = self.vstar_cache.get(prompt, self.config)
            
            if cached_data is not None:
                # Cache hit!
                V_star_values.append(cached_data['v_star'])
                cache_hits += 1
            else:
                # Cache miss - need to compute
                prompts_to_compute.append(prompt)
                prompt_indices.append(i)
                V_star_values.append(None)  # Placeholder
        
        # ===== SECOND PASS: BATCH COMPUTE UNCACHED PROMPTS =====
        if prompts_to_compute:
            print(f"  Generating samples for {len(prompts_to_compute)} uncached prompts...")
            
            # Generate all samples (batched for efficiency)
            all_samples = []
            for prompt in prompts_to_compute:
                samples = self.ref_model.generate(
                    [prompt],
                    num_samples=num_samples,
                    min_new_tokens=30,
                    temperature=1.0,
                    max_length=self.gen_max_length
                )[0]
                all_samples.append(samples)
            
            # Compute V* for each uncached prompt
            for idx, prompt in enumerate(prompts_to_compute):
                samples = all_samples[idx]
                original_idx = prompt_indices[idx]
                problem = problems[original_idx] if problems else {'prompt': prompt, 'task': 'unknown'}
                
                # Compute rewards (binary for V*)
                rewards = [compute_reward(s, problem, require_cot=False) for s in samples]
                rewards = np.array(rewards, dtype=np.float32)
                
                # Calculate V*
                if rewards.size == 0:
                    V_star = 0.0
                else:
                    if self.beta > 0:
                        max_r = rewards.max()
                        exp_terms = np.exp((rewards - max_r) / self.beta)
                        V_star = float(max_r + self.beta * np.log(np.mean(exp_terms)))
                    else:
                        V_star = float(rewards.max())
                
                # Save to cache
                cache_data = {
                    'v_star': V_star,
                    'rewards': rewards.tolist(),
                    'num_samples': len(samples)
                }
                self.vstar_cache.save(prompt, self.config, cache_data)
                
                V_star_values[original_idx] = V_star
        
        # ===== PRINT CACHE STATS =====
        if cache_hits > 0:
            hit_rate = cache_hits / len(prompts) * 100
            print(f"  V* cache: {cache_hits}/{len(prompts)} hits ({hit_rate:.1f}% - saved compute!)")
        
        return np.array(V_star_values, dtype=np.float32)


    def _build_concat_with_labels(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, pad_id: int):
        """Construct input_ids and labels with prompt masking."""
        device = prompt_ids.device
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.clone()
        labels[:] = -100 # Mask all initially

        prompt_lens = (prompt_ids != pad_id).sum(dim=1)
        B, T = labels.size()
        for i in range(B):
            start = int(prompt_lens[i].item())
            # Only unmask if start index is within sequence length
            if start < T:
                labels[i, start:] = input_ids[i, start:]

        return input_ids, attention_mask, labels

    def _per_example_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean CE per example over unmasked tokens."""
        B, T, V = logits.shape
        # Shift logits and labels for next token prediction loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        # Calculate loss per token, ignore pad index
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='none'
        ).view(B, T - 1) # Reshape back to [B, T-1]

        # Mask based on shifted labels
        token_mask = (shift_labels != -100).float()
        # Calculate mean loss per example
        per_ex_loss = (token_losses.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0))

        # Handle cases where an example has no valid labels (e.g., prompt filled max_length)
        per_ex_loss = torch.nan_to_num(per_ex_loss, nan=0.0) # Replace NaN with 0 if no valid tokens

        return per_ex_loss


    def _compute_kl_loss(self, logits_pi: torch.Tensor, logits_ref: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes KL divergence per example only on completion tokens """
        # Shift logits and labels like in CE loss
        logits_pi_shifted = logits_pi[..., :-1, :].contiguous()
        logits_ref_shifted = logits_ref[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        logp_pi = F.log_softmax(logits_pi_shifted, dim=-1)
        logp_ref = F.log_softmax(logits_ref_shifted, dim=-1)

        # Mask for completion tokens (labels != -100)
        token_mask = (labels_shifted != -100).float()

        # KL divergence per token: sum_vocab p_pi * (logp_pi - logp_ref)
        # Use formula: E_{token ~ pi} [log p_pi(token) - log p_ref(token)]
        # We approximate this with the sampled token's contribution
        # Need predicted probabilities, not just logprobs at true labels
        kl_div_tokens = F.kl_div(logp_ref, logp_pi, log_target=True, reduction='none').sum(-1) # sum over vocab
        kl_div_tokens = kl_div_tokens * token_mask # Apply mask

        # Average KL per example over valid tokens
        kl_per_ex = kl_div_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        kl_per_ex = torch.nan_to_num(kl_per_ex, nan=0.0) # Handle NaN

        return kl_per_ex


    def train_step(self, batch: List[Dict]) -> tuple:
        """A*PO training step with stable advantage weighting"""
        prompts = [item['prompt'] for item in batch]
        # Handle DDP wrapping - get actual device
        actual_model = self.policy.model.module if hasattr(self.policy.model, 'module') else self.policy.model
        device = actual_model.device

        try:
            # Step 1: Compute V*
            V_star_np = self.compute_V_star(prompts, problems=batch)
            V_star_t = torch.tensor(V_star_np, dtype=torch.float32, device=device)

            # Step 2: Generate from policy
            # CRITICAL FIX: Set to eval mode for generation with DDP+gradient checkpointing
            # Training mode with gradient checkpointing causes corrupted outputs in DDP!
            self.policy.eval()  # ← FIXED: eval mode for clean generation
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            generated_texts = self.policy.generate(
                prompts,
                max_length=self.gen_max_length,
                min_new_tokens=30,
                temperature=self.temperature,
                do_sample=True,
                top_p=self.top_p,
                top_k=self.top_k
            )

            # Debug logging on first step to verify generation quality
            if self.step == 0:
                print("\nDEBUG: First policy generation sample:")
                print(f"   Prompt: {prompts[0][:80]}...")
                print(f"   Generated: {generated_texts[0][:200]}...")
                print()

            # Step 3: Compute rewards
            # Use standard reward function (math tasks)
            rewards = [compute_reward(text, problem, require_cot=False)
                   for text, problem in zip(generated_texts, batch)]
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Set back to train mode for forward pass
            self.policy.train()

            # Step 4: Compute advantages & weights (STABLE VERSION)
            advantages = rewards_t - V_star_t # [B]

            # Normalize advantages across the batch for stability
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-6 # Avoid division by zero
            adv_norm = (advantages - adv_mean) / adv_std
            adv_norm = adv_norm.clamp(-self.adv_clip, self.adv_clip).detach() # Clip and detach

            # --- CHOOSE WEIGHTING SCHEME ---
            if self.weighting_scheme == 'exp':
                # Original A* weighting (can be unstable)
                weights = torch.exp(advantages / (self.beta + 1e-8)).detach()
                # Normalize weights to mean 1.0 (helps stabilize LR)
                weights = weights / weights.mean().clamp_min(1e-6)
            elif self.weighting_scheme == 'shifted_advantage':
                # Shift normalized advantages to be non-negative (often more stable)
                weights = (adv_norm + self.adv_clip).detach() # Shifts range to [0, 2*adv_clip]
                # Optional: Normalize to mean 1? Might not be necessary if shifted.
                # weights = weights / weights.mean().clamp_min(1e-6)
            else: # Default: 'normalized_advantage'
                # Use clipped normalized advantages directly (Simplest, often stable)
                # Adding 1.0 shifts the center from 0 to 1, range becomes approx [-2, 4] if clip=3
                # Clamping > 0 avoids negative loss contributions if CE is always positive
                weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0).detach()

            # --- Logging Intermediate Values ---
            if self.log_intermediate_values and self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print("\n--- Intermediate Values ---")
                for i in range(len(prompts)):
                    print(f"  Ex {i}: Reward={rewards_t[i]:.3f}, V*={V_star_t[i]:.3f}, Adv={advantages[i]:.3f}, AdvNorm={adv_norm[i]:.3f}, Weight={weights[i]:.3f}")
                print(f"  Advantage Stats: Mean={adv_mean:.3f}, Std={adv_std:.3f}")
                print(f"  Weight Stats: Mean={weights.mean():.3f}, Std={weights.std():.3f}, Min={weights.min():.3f}, Max={weights.max():.3f}")
                print("-------------------------\n")


            # Step 5: Build teacher-forced training batch
            enc_prompts = self.policy.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.sft_max_length)
            # Tokenize completions *without* special tokens if they match prompt start/end issues
            enc_comps = self.policy.tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.sft_max_length, add_special_tokens=False) # Try disabling special tokens for completion
            enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
            enc_comps = {k: v.to(device) for k, v in enc_comps.items()}

            pad_id = self.policy.tokenizer.pad_token_id or getattr(self.policy.tokenizer, "eos_token_id", 0)

            input_ids, attention_mask, labels = self._build_concat_with_labels(
                enc_prompts["input_ids"], enc_comps["input_ids"], pad_id
            )

            # Truncate to sft_max_length
            input_ids = input_ids[:, :self.sft_max_length]
            attention_mask = attention_mask[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # Step 6: Forward pass (policy)
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None # Compute custom loss
            )
            policy_logits = outputs.logits
            per_ex_ce_loss = self._per_example_ce_loss(policy_logits, labels) # [B]

            # Step 7: KL divergence term (if enabled)
            kl_term = torch.zeros_like(per_ex_ce_loss) # [B]
            if self.kl_coef and self.kl_coef > 0.0 and hasattr(self.ref_model, "model"):
                with torch.no_grad():
                    self.ref_model.model.eval() # Ensure ref model is in eval mode
                    ref_outputs = self.ref_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                ref_logits = ref_outputs.logits.detach()
                kl_per_ex = self._compute_kl_loss(policy_logits, ref_logits, labels) # [B]
                kl_term = self.kl_coef * kl_per_ex

                # Add KL per-example to loss
                per_ex_loss_with_kl = per_ex_ce_loss + kl_term
            else:
                per_ex_loss_with_kl = per_ex_ce_loss

            # Step 8: Weight per-example loss and reduce
            # Ensure weights don't have NaNs/Infs
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                 print("Warning: NaN or Inf detected in weights, using uniform weights for this step.")
                 weights = torch.ones_like(weights)

            # Apply weights (already detached)
            weighted_losses = per_ex_loss_with_kl * weights
            loss = weighted_losses.mean() # Average over batch

            # Check for NaN/Inf loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print("Error: NaN or Inf loss detected before backward pass. Skipping step.")
                print(f"Loss: {loss.item()}, Weights: {weights}, PerExLoss: {per_ex_loss_with_kl}")
                # Potentially log more details here
                raise ValueError("NaN/Inf loss detected") # Stop training

            # Step 9: Backprop & update
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Metrics
            self.step += 1
            loss_value = float(loss.item())
            avg_reward = float(rewards_t.mean().item())
            avg_advantage = float(advantages.mean().item()) # Raw advantage
            avg_v_star = float(V_star_t.mean().item())
            avg_kl = float(kl_term.mean().item()) # Average KL penalty per example


            if self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print(
                    f"Step {self.step}: "
                    f"Loss={loss_value:.4f} (CE={per_ex_ce_loss.mean().item():.4f}, KL={avg_kl:.4f}), "
                    f"Reward={avg_reward:.3f}, "
                    f"Advantage={avg_advantage:.3f}, "
                    f"V*={avg_v_star:.3f}"
                )

            stats = {
                'loss': loss_value,
                'avg_reward': avg_reward,
                'avg_advantage': avg_advantage,
                'avg_v_star': avg_v_star,
                'avg_kl_penalty': avg_kl,
                'adv_norm_mean': float(adv_norm.mean().item()), # Should be near 0
                'adv_norm_std': float(adv_norm.std().item()),   # Should be near 1 before clipping
                'weight_mean': float(weights.mean().item()),
                'weight_std': float(weights.std().item()),
            }

            return loss_value, stats

        except Exception as e:
            print(f"\n--- Error in train_step ---")
            print(f"Step: {self.step}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            print("---------------------------\n")

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            # Return safe default values
            return 0.0, {
                'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_v_star': 0.0, 'avg_kl_penalty': 0.0, 'adv_norm_mean': 0.0,
                'adv_norm_std': 0.0, 'weight_mean': 1.0, 'weight_std': 0.0,
            }