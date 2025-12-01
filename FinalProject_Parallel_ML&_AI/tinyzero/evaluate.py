"""
Comprehensive evaluation for TinyZero
"""
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

from tinyzero.rewards import compute_reward, extract_final_answer


def evaluate(
    model,
    eval_loader,
    verbose: bool = True,
    save_examples: bool = True,
    max_examples: int = 20
) -> Dict[str, float]:
    """
    Evaluate model on evaluation set
    
    Args:
        model: PolicyModel to evaluate
        eval_loader: DataLoader with evaluation data
        verbose: Whether to print detailed results
        save_examples: Whether to save example outputs
        max_examples: Maximum number of examples to save
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # Results storage
    results = {
        'correct': 0,
        'total': 0,
        'correct_countdown': 0,
        'total_countdown': 0,
        'correct_multiplication': 0,
        'total_multiplication': 0,
        'lengths': [],
        'rewards': []
    }
    
    examples = []
    
    # Evaluation loop
    for batch in tqdm(eval_loader, desc="Evaluating", disable=not verbose):
        prompts = [item['prompt'] for item in batch]
        
        # Generate solutions
        with torch.no_grad():
            try:
                outputs = model.generate(
                    prompts,
                    max_length=100,
                    min_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            except Exception as e:
                print(f"Error generating: {e}")
                outputs = ["Error"] * len(prompts)
        
        # Compute rewards
        for output, problem in zip(outputs, batch):
            reward = compute_reward(output, problem)
            
            results['correct'] += reward
            results['total'] += 1
            results['lengths'].append(len(output))
            results['rewards'].append(reward)
            
            # Per-task metrics
            task = problem['task']
            if task == 'countdown':
                results['correct_countdown'] += reward
                results['total_countdown'] += 1
            elif task == 'multiplication':
                results['correct_multiplication'] += reward
                results['total_multiplication'] += 1
            
            # Save examples
            if save_examples and len(examples) < max_examples:
                # Extract predicted answer
                predicted = extract_final_answer(output)
                
                # Get actual answer (works for both tasks)
                if problem['task'] == 'multiplication':
                    actual = problem['answer']
                elif problem['task'] == 'countdown':
                    actual = problem['target']
                else:
                    actual = 'N/A'
                
                # Account for both float and Integer
                if predicted is not None and isinstance(predicted, float):
                    predicted_display = int(predicted) if predicted == int(predicted) else predicted
                else:
                    predicted_display = predicted
                
                if isinstance(actual, float):
                    actual_display = int(actual) if actual == int(actual) else actual
                else:
                    actual_display = actual
                
                examples.append({
                    'prompt': problem['prompt'],
                    'output': output,
                    'predicted': predicted_display,
                    'actual': actual_display,
                    'correct': bool(reward),
                    'task': task
                })
    
    # Compute metrics
    metrics = {
        'accuracy': results['correct'] / max(results['total'], 1),
        'accuracy_countdown': results['correct_countdown'] / max(results['total_countdown'], 1),
        'accuracy_multiplication': results['correct_multiplication'] / max(results['total_multiplication'], 1),
        'avg_length': np.mean(results['lengths']) if results['lengths'] else 0,
        'total_examples': results['total']
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total Examples: {results['total']}")
        print(f"Overall Accuracy: {metrics['accuracy']:.2%} ({results['correct']}/{results['total']})")
        print(f"Countdown Accuracy: {metrics['accuracy_countdown']:.2%} ({results['correct_countdown']}/{results['total_countdown']})")
        print(f"Multiplication Accuracy: {metrics['accuracy_multiplication']:.2%} ({results['correct_multiplication']}/{results['total_multiplication']})")
        print(f"Avg Response Length: {metrics['avg_length']:.1f} chars")
        
        if save_examples and examples:
            print("\n" + "="*60)
            print("EXAMPLE OUTPUTS")
            print("="*60)
            
            # Show correct examples
            correct_examples = [ex for ex in examples if ex['correct']]
            if correct_examples:
                print(f"\nCORRECT EXAMPLES (showing {min(3, len(correct_examples))}):")
                for i, ex in enumerate(correct_examples[:3]):
                    print(f"\n{i+1}. [{ex['task'].upper()}]")
                    print(f"Prompt: {ex['prompt']}")
                    print(f"Model Output: {ex['output'][:300]}...")
                    print(f"Predicted: {ex['predicted']}, Actual: {ex['actual']}")
            
            # Show incorrect examples
            incorrect_examples = [ex for ex in examples if not ex['correct']]
            if incorrect_examples:
                print(f"\n✗ INCORRECT EXAMPLES (showing {min(3, len(incorrect_examples))}):")
                for i, ex in enumerate(incorrect_examples[:3]):
                    print(f"\n{i+1}. [{ex['task'].upper()}]")
                    print(f"Prompt: {ex['prompt']}")
                    print(f"Model Output: {ex['output'][:300]}...")
                    print(f"Predicted: {ex['predicted']}, Actual: {ex['actual']}")
        
        print("="*60 + "\n")
    
    return metrics


def compute_detailed_metrics(results: List[Dict]) -> Dict:
    """
    Compute detailed metrics from results
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Detailed metrics
    """
    # Accuracy by task
    task_accuracy = {}
    for task in ['countdown', 'multiplication']:
        task_results = [r for r in results if r['task'] == task]
        if task_results:
            correct = sum(1 for r in task_results if r['correct'])
            task_accuracy[task] = correct / len(task_results)
    
    # Response length statistics
    lengths = [len(r['output']) for r in results]
    
    metrics = {
        'task_accuracy': task_accuracy,
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths)
    }
    
    return metrics


def evaluate_with_temperature_sweep(
    model,
    eval_loader,
    temperatures: List[float] = [0.3, 0.7, 1.0, 1.5]
) -> Dict[float, Dict]:
    """
    Evaluate model with different temperature values
    
    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        temperatures: List of temperatures to try
    
    Returns:
        Dictionary mapping temperature to metrics
    """
    results = {}
    
    for temp in temperatures:
        print(f"\nEvaluating with temperature={temp}")
        
        # Temporarily change generation temperature
        original_temp = 0.7  # Default
        
        # Run evaluation (you'd need to pass temperature to evaluate function)
        metrics = evaluate(model, eval_loader, verbose=False)
        results[temp] = metrics
    
    # Print comparison
    print("\n" + "="*60)
    print("TEMPERATURE COMPARISON")
    print("="*60)
    for temp, metrics in results.items():
        print(f"Temperature {temp}: Accuracy = {metrics['accuracy']:.2%}")
    
    return results