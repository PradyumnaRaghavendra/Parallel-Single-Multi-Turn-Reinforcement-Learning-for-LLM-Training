"""
Model wrappers for TinyZero
Robust generation: prefer max_new_tokens, avoid max_length conflicts.
"""
from typing import List, Optional, Dict, Any
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _pick_device(requested: str) -> str:
    # Respect explicit device requests
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    # Fallback: try MPS, then CPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_padding(tokenizer):
    # Many models lack a pad token; using eos as pad is common for causal LMs
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding tends to be safer for batched causal LMs
    tokenizer.padding_side = "left"


def _sanitize_gen_kwargs(
    tokenizer,
    gen_kwargs: Dict[str, Any],
    default_max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    - Prefer max_new_tokens.
    - If both max_length and max_new_tokens are present, drop max_length.
    - If neither present, set max_new_tokens to a safe default.
    - Ensure pad/eos token ids exist.
    """
    gen = dict(gen_kwargs) if gen_kwargs else {}

    # If both are present, remove max_length to avoid HF warnings/errors
    if "max_new_tokens" in gen and "max_length" in gen:
        gen.pop("max_length")

    # If neither specified, set a safe default for continuation length
    if "max_new_tokens" not in gen and "max_length" not in gen:
        gen["max_new_tokens"] = default_max_new_tokens

    # Reasonable defaults (can be overridden by caller)
    gen.setdefault("do_sample", True)
    gen.setdefault("pad_token_id", tokenizer.pad_token_id)
    gen.setdefault("eos_token_id", tokenizer.eos_token_id)

    return gen


class PolicyModel:
    """Wrapper for the trainable policy model."""

    def __init__(self, model_name: str, device: str = "cuda", max_input_len: int = 2048, gradient_checkpointing: bool = False):
        self.device = _pick_device(device)
        self.max_input_len = max_input_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_padding(self.tokenizer)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,     # keep it simple & compatible
            low_cpu_mem_usage=True,
        ).to(self.device)

        # Enable gradient checkpointing to save memory if requested
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Train loop can call .train(); default to eval for safe generation
        self.model.eval()

    def _get_model(self):
        """Get unwrapped model (handles DDP/FSDP wrapping)"""
        # Check if wrapped in DDP or FSDP
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_length: int = 2048,                 # kept for backward compatibility (prompt cap)
        min_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **extra_gen_kwargs,
    ) -> List[str]:
        """
        Robust generation:
          - Truncates *prompt* to max_input_len (NOT used as output cap).
          - Prefers `max_new_tokens` and removes conflicting `max_length`.
          - Returns only the continuation (prompt stripped).
        """
        # Tokenize prompts with a cap on prompt length (avoid OOM)
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(max_length, self.max_input_len),
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        # Build generation kwargs safely
        gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if min_new_tokens is not None and min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if top_p is not None and top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            gen_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty

        # Merge caller extras and sanitize (prefer max_new_tokens)
        gen_kwargs.update(extra_gen_kwargs)
        gen_kwargs = _sanitize_gen_kwargs(self.tokenizer, gen_kwargs)

        # Generate (use unwrapped model to handle DDP/FSDP)
        outputs = self._get_model().generate(
            input_ids=input_ids,
            attention_mask=attn,
            **gen_kwargs,
        )

        # Return ONLY the continuation (strip prompt)
        prompt_len = input_ids.shape[1]
        out_texts: List[str] = []
        for i in range(outputs.size(0)):
            cont_ids = outputs[i, prompt_len:]
            text = self.tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
            out_texts.append(text)
        return out_texts

    def get_log_probs(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """
        Returns average log prob over the completion tokens (simple baseline).
        Shape: (batch,)
        """
        assert len(prompts) == len(completions)
        # Tokenize prompt and full text to get completion span
        enc_prompt = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_len
        )
        enc_full = self.tokenizer(
            [p + c for p, c in zip(prompts, completions)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        )

        input_ids = enc_full["input_ids"].to(self.device)
        attn = enc_full["attention_mask"].to(self.device)
        prompt_lens = (enc_prompt["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            logp = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_lp = []
        for b in range(input_ids.size(0)):
            # tokens corresponding to the completion
            start = int(prompt_lens[b].item()) - 1  # next-token prediction starts at last prompt token
            start = max(start, 0)
            # indices [start .. last-1] predict tokens [start+1 .. last]
            tok_ids = input_ids[b, start + 1 :]
            tok_logits = logp[b, start : start + tok_ids.size(0), :]
            tok_lp = tok_logits.gather(-1, tok_ids.unsqueeze(-1)).squeeze(-1)
            if tok_lp.numel() == 0:
                batch_lp.append(torch.tensor(0.0, device=self.device))
            else:
                batch_lp.append(tok_lp.mean())
        return torch.stack(batch_lp, dim=0)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class ReferenceModel(PolicyModel):
    """Frozen reference model for computing V* or baselines."""

    def __init__(self, model_name: str, device: str = "cuda", max_input_len: int = 2048):
        super().__init__(model_name, device=device, max_input_len=max_input_len)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,                 # prompt cap for ref sampling
        min_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        num_samples: int = 1,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **extra_gen_kwargs,
    ) -> List[List[str]]:
        """
        Generate multiple samples per prompt.
        Uses the same sanitization logic (prefer max_new_tokens).
        """
        all_out: List[List[str]] = []

        for prompt in prompts:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(max_length, self.max_input_len),
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)

            gen_kwargs: Dict[str, Any] = dict(
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if min_new_tokens is not None and min_new_tokens > 0:
                gen_kwargs["min_new_tokens"] = min_new_tokens
            if top_p is not None and top_p < 1.0:
                gen_kwargs["top_p"] = top_p
            if top_k is not None and top_k > 0:
                gen_kwargs["top_k"] = top_k
            gen_kwargs.update(extra_gen_kwargs)
            gen_kwargs = _sanitize_gen_kwargs(self.tokenizer, gen_kwargs)

            samples: List[str] = []
            for _ in range(max(1, num_samples)):
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **gen_kwargs,
                )
                cont_ids = out[0, input_ids.shape[1] :]
                text = self.tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
                samples.append(text)
            all_out.append(samples)

        return all_out


if __name__ == "__main__":
    print("Testing model loading…")
    policy = PolicyModel("gpt2", device="cpu")
    prompt = ["What is 5 + 3?"]
    # no max_length/max_new_tokens passed -> defaults to max_new_tokens=64
    resp = policy.generate(prompt, temperature=0.7, top_p=0.9)
    print("Prompt:", prompt[0])
    print("Response:", resp[0][:200])
    print("OK.")
