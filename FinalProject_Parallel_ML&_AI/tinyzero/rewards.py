print("="*60)
print("REWARDS MODULE LOADED - VERSION 3.0 WITH PROPER WEBSHOP REWARDS")
print("="*60)
"""
Unified reward system for TinyZero and RAGEN
FIXED: Proper WebShop reward computation based on trajectory quality
"""
import re
from typing import Dict, Any, Optional, Union, List
import math


# ============================================================
# MATH REWARDS (TinyZero Original) - UNCHANGED
# ============================================================

def extract_final_answer(text: str) -> Optional[float]:
    """Extract numerical answer from model output."""
    answer_tag = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    if answer_tag:
        answer_str = answer_tag.group(1).replace(",", "").strip()
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', answer_str)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
    
    boxed_match = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed_match:
        answer_str = boxed_match.group(1).replace(",", "").strip()
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    patterns = [
        r"(?:final answer is|the answer is|result is|equals?)\s*:?\s*(-?[\d,]+(?:\.\d+)?)",
        r"=\s*(-?[\d,]+(?:\.\d+)?)\s*(?:[\.\?!]|$)",
    ]
    
    for pattern in reversed(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            answer_str = matches[-1].replace(",", "").strip()
            try:
                return float(answer_str)
            except ValueError:
                continue
    
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        last_num_str = numbers[-1].replace(",", "").strip()
        try:
            return float(last_num_str)
        except ValueError:
            pass
    
    return None


def compute_math_reward(
    generated_text: str,
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    check_reasoning: bool = False
) -> float:
    """Compute reward for math problems."""
    predicted_answer_num = extract_final_answer(generated_text)
    
    if predicted_answer_num is None:
        return 0.0
    
    if problem.get('task') == 'multiplication':
        correct_answer = problem['answer']
        if math.isclose(predicted_answer_num, correct_answer, abs_tol=0.01):
            return 1.0
    
    elif problem.get('task') == 'countdown':
        target = problem['target']
        if target != 0 and math.isclose(predicted_answer_num, target, rel_tol=tolerance):
            return 1.0
        elif target == 0 and math.isclose(predicted_answer_num, target, abs_tol=tolerance):
            return 1.0
    
    return 0.0


# ============================================================
# WEBSHOP REWARDS (RAGEN Extension) - COMPLETELY REWRITTEN
# ============================================================

def parse_webshop_action(action: str) -> tuple:
    """
    Parse a WebShop action to extract type and argument.
    
    Returns:
        (action_type, argument) tuple
        Examples:
        - "search[blue headphones]" → ("search", "blue headphones")
        - "click[B09QKP7XQL]" → ("click", "B09QKP7XQL")
        - "buy now" → ("buy", "")
    """
    action = str(action).strip().lower()
    
    # Try bracket format: action[argument]
    match = re.match(r'(\w+)\[(.*?)\]', action)
    if match:
        return (match.group(1), match.group(2))
    
    # Simple commands
    if 'buy' in action:
        return ("buy", "")
    if 'back' in action:
        return ("back", "")
    
    return ("unknown", action)


def is_valid_webshop_action(action: str) -> bool:
    """
    Check if action follows WebShop format.
    Valid actions:
    - search[query]
    - click[product_id]
    - buy now
    - back
    """
    action = str(action).strip().lower()
    
    # Check for valid formats
    valid_patterns = [
        r'^search\[.+\]$',       # search[something]
        r'^click\[[\w-]+\]$',    # click[product_id]
        r'^buy(\s+now)?$',       # buy or buy now
        r'^back$',               # back
    ]
    
    return any(re.match(pattern, action) for pattern in valid_patterns)


def compute_webshop_reward(
    trajectory: Dict[str, Any],
    task: Dict[str, Any],
    step: int = 0
) -> float:
    """
    EXPERT-LEVEL reward shaping combining Options A & B.

    Key improvements:
    1. EXPLICIT rewards for correct action sequences
    2. STRONG penalties for wrong/invalid actions
    3. Progressive bonuses that guide toward success
    4. Format validation with immediate feedback
    """
    if not trajectory or not isinstance(trajectory, dict):
        return 0.0

    actions = trajectory.get('actions', [])
    if not actions:
        return 0.0

    env_reward = float(trajectory.get('total_reward', 0.0))

    # ========================================
    # CASE 1: WebShop Success - MAXIMUM REWARD
    # ========================================
    if env_reward >= 0.9:
        return 1.0  # Full success
    elif env_reward >= 0.5:
        return 0.8  # Partial success (correct product, wrong attributes)
    elif env_reward > 0.0:
        return 0.5 + env_reward  # Some reward given

    # ========================================
    # CASE 2: Enhanced Dense Reward Shaping
    # ========================================
    reward = 0.0

    # Extract task info for relevance checking
    task_text = task.get('instruction', '').lower()
    task_words = set(task_text.split())

    # Track trajectory quality
    searched = False
    clicked = False
    bought = False
    relevant_search = False
    valid_format_count = 0
    invalid_format_count = 0

    # ========================================
    # STEP 1: Score Each Action (with penalties)
    # ========================================
    for action in actions:
        action_lower = str(action).lower().strip()

        # Validate format FIRST
        is_valid = (
            (action_lower.startswith('search[') and ']' in action_lower) or
            (action_lower.startswith('click[') and ']' in action_lower) or
            action_lower in ['buy now', 'buy', 'back']
        )

        if is_valid:
            valid_format_count += 1
        else:
            invalid_format_count += 1
            reward -= 0.15  # ★ STRONG PENALTY for invalid format
            continue  # Skip this action

        # Search action - reward relevance
        if action_lower.startswith('search['):
            searched = True
            try:
                search_query = action_lower.split('search[')[1].split(']')[0]
                search_words = set(search_query.split())
                overlap = len(task_words & search_words)

                if overlap >= 2:
                    relevant_search = True
                    reward += 0.25  # ★ STRONG reward for relevant search
                elif overlap >= 1:
                    reward += 0.10  # Partial credit
                else:
                    reward -= 0.05  # ★ Small penalty for irrelevant search
            except:
                reward -= 0.10  # ★ Penalty for malformed search

        # Click action - always good progress
        elif action_lower.startswith('click['):
            clicked = True
            reward += 0.20  # Good progress

            # Verify it's a valid product ID (B0XXXXXXXX format)
            try:
                product_id = action_lower.split('click[')[1].split(']')[0]
                if product_id.startswith('b0') or product_id.startswith('b1'):
                    reward += 0.05  # Bonus for valid format
            except:
                pass

        # Buy action - final step
        elif 'buy' in action_lower:
            bought = True
            reward += 0.25  # Attempting to complete

    # ========================================
    # STEP 2: Sequence Bonuses (Explicit Learning)
    # ========================================

    # Base exploration bonus
    if searched:
        reward += 0.10  # Encourage searching

    # Good sequence: search → click (model found product)
    if searched and clicked:
        reward += 0.20  # ★ STRONG bonus for correct flow

    # Better sequence: relevant search → click
    if relevant_search and clicked:
        reward += 0.15  # Extra bonus for relevance

    # Great sequence: click → buy (model attempting completion)
    if clicked and bought:
        reward += 0.25  # ★ STRONG bonus - almost perfect!

    # Perfect sequence: search → click → buy
    if searched and clicked and bought:
        reward += 0.20  # ★ Maximum sequence bonus

    # ========================================
    # STEP 3: Penalties for Bad Behavior
    # ========================================

    # Penalty for NO valid actions
    if valid_format_count == 0:
        reward -= 0.30  # ★ STRONG penalty - model learned nothing

    # Penalty for excessive back/repetition
    back_count = sum(1 for a in actions if 'back' in str(a).lower())
    if back_count > 2:
        reward -= 0.10 * (back_count - 2)  # ★ Escalating penalty

    # Penalty for buying without clicking (blind buying)
    if bought and not clicked:
        reward -= 0.25  # ★ STRONG penalty - wrong sequence

    # Penalty for clicking without searching
    if clicked and not searched:
        reward -= 0.15  # ★ Penalty - skipped search step

    # Penalty for too many actions (inefficiency)
    if len(actions) > 8:
        reward -= 0.05 * (len(actions) - 8)  # Gentle penalty for verbosity

    # ========================================
    # STEP 4: Curriculum Bonus (if available)
    # ========================================
    try:
        from ragen.curriculum_rewards import curriculum_reward_bonus
        curriculum_bonus = curriculum_reward_bonus(trajectory, step)
        reward += curriculum_bonus
    except:
        pass

    # ========================================
    # RETURN: Clamp to valid range
    # ========================================
    # Keep reward below 1.0 (reserved for true success)
    # Allow small negatives to signal bad behavior
    return max(-0.2, min(reward, 0.85))


# ============================================================
# UNIFIED REWARD FUNCTION (Auto-Detect)
# ============================================================

def compute_reward(
    generated_output: Union[str, Dict[str, Any]],
    problem: Dict[str, Any],
    tolerance: float = 0.01,
    require_cot: bool = False,
    step: int = 0
) -> float:
    """
    Unified reward function - auto-detects environment type.
    
    Supports:
    - TinyZero math tasks (text input)
    - RAGEN WebShop tasks (trajectory dict input)
    
    Args:
        generated_output: Either text string (math) or trajectory dict (WebShop)
        problem: Problem/task specification
        tolerance: Tolerance for numerical answers
        require_cot: Whether to require chain-of-thought (math only)
    
    Returns:
        Reward value (0.0 to 1.0)
    """
    # Auto-detect based on input type
    if isinstance(generated_output, dict):
        # Dictionary with trajectory info → WebShop
        if 'actions' in generated_output or 'total_reward' in generated_output:
            return compute_webshop_reward(generated_output, problem, step)
    
    # String output → Math task
    if isinstance(generated_output, str):
        return compute_math_reward(generated_output, problem, tolerance, require_cot)
    
    # Unknown type
    print(f"Warning: Unknown output type for reward: {type(generated_output)}")
    return 0.0


# def compute_reward_with_partial_credit(
#     generated_text: str,
#     problem: Dict[str, Any],
#     tolerance: float = 0.01,
#     check_reasoning: bool = True
# ) -> float:
#     """
#     Compute reward with PARTIAL CREDIT for reasoning.
#     (For TinyZero math tasks only - kept for compatibility)
#     """
#     # For now, just use binary rewards
#     return compute_math_reward(generated_text, problem, tolerance, False)


# ============================================================
# TESTING / DEBUGGING HELPERS
# ============================================================

def debug_webshop_reward(trajectory: Dict[str, Any], task: Dict[str, Any]) -> None:
    """Print detailed reward breakdown for debugging."""
    print("\n" + "="*60)
    print("REWARD DEBUG")
    print("="*60)
    
    actions = trajectory.get('actions', [])
    env_reward = trajectory.get('total_reward', 0.0)
    
    print(f"Environment Reward: {env_reward:.3f}")
    print(f"Number of actions: {len(actions)}")
    print(f"\nActions:")
    for i, action in enumerate(actions):
        action_type, arg = parse_webshop_action(action)
        is_valid = is_valid_webshop_action(action)
        print(f"  {i+1}. {action[:60]}")
        print(f"     → Type: {action_type}, Valid: {is_valid}")
    
    computed_reward = compute_webshop_reward(trajectory, task)
    print(f"\nComputed Reward: {computed_reward:.3f}")
    print("="*60 + "\n")


# Export functions
__all__ = [
    'compute_reward',
    'compute_reward_with_partial_credit',
    'compute_math_reward',
    'compute_webshop_reward',
    'extract_final_answer',
    'is_valid_webshop_action',
    'parse_webshop_action',
    'debug_webshop_reward',
]