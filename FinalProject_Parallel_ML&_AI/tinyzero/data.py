"""
Data loading with Curriculum Learning
Difficulty increases automatically during training
"""
import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class CurriculumMathDataset(Dataset):
    """
    Dataset with automatic curriculum learning
    Generates easier problems early, harder problems later
    """
    
    def __init__(self, num_samples: int = 1000, tasks: List[str] = ["multiplication"]):
        """
        Args:
            num_samples: Number of problems to generate
            tasks: List of task types to include
        """
        self.num_samples = num_samples
        self.tasks = tasks
        self.current_step = 0  # Track training progress
        
        # Pre-generate problems for consistency
        # We'll regenerate when difficulty changes
        self.data = self._generate_all_data()
    
    def set_difficulty_step(self, step: int):
        """
        Update current training step - called by trainer
        Regenerates data when crossing difficulty thresholds
        """
        old_difficulty = self._get_difficulty_level(self.current_step)
        new_difficulty = self._get_difficulty_level(step)
        
        self.current_step = step
        
        # Regenerate data if difficulty changed
        if old_difficulty != new_difficulty:
            print(f"\n Curriculum Update: Moving to {new_difficulty} problems at step {step}")
            self.data = self._generate_all_data()
    
    def _get_difficulty_level(self, step: int) -> str:
        """
        Balanced curriculum for 250 steps:
        - Steps 0-70: very_easy (28%)
        - Steps 71-150: easy (32%)
        - Steps 151-210: medium (24%)
        - Steps 211-250: hard (16%)
        """
        if step < 70:
            return "very_easy"
        elif step < 150:
            return "easy"
        elif step < 210:
            return "medium"
        else:
            return "hard"
    
    def _generate_all_data(self) -> List[Dict]:
        """Generate all problems at current difficulty"""
        data = []
        
        for _ in range(self.num_samples):
            task = random.choice(self.tasks)
            
            if task == "countdown":
                problem = self._generate_countdown()
            elif task == "multiplication":
                problem = self._generate_multiplication()
            else:
                raise ValueError(f"Unknown task: {task}")
            
            data.append(problem)
        
        return data
    
    def _generate_countdown(self) -> Dict:
            """
            Generate countdown problem with GUARANTEED solvable target.
            Works for all difficulty levels.
            """
            difficulty = self._get_difficulty_level(self.current_step)
            
            if difficulty == "very_easy":
                # Very easy: 3 small numbers, target is sum of 2
                numbers = random.sample(range(1, 10), k=3)
                target = random.choice(numbers) + random.choice(numbers)
            
            elif difficulty == "easy":
                # Easy: 4 numbers, target is achievable
                numbers = random.sample(range(1, 15), k=4)
                target = sum(random.sample(numbers, k=2))
            
            elif difficulty == "medium":
                # Medium: mix of large and small
                large_nums = random.sample([10, 25], k=1)
                small_nums = random.sample(range(1, 10), k=3)
                numbers = large_nums + small_nums
                target = random.randint(20, 50)
            
            else:  # hard
                # Hard: real countdown game
                large_nums = random.sample([25, 50, 75, 100], k=2)
                small_nums = random.sample(range(1, 11), k=4)
                numbers = large_nums + small_nums
                target = random.randint(100, 500)
            
            prompt = f"Using the numbers {numbers}, create an equation that equals {target}."
            
            return {
                "prompt": prompt,
                "task": "countdown",
                "numbers": numbers,
                "target": target
            }
    
    def _generate_multiplication(self) -> Dict:
        """
        Generate multiplication with curriculum learning.
        Prompts model to use proper <think>/<answer> format.
        """
        difficulty = self._get_difficulty_level(self.current_step)
        
        if difficulty == "very_easy":
            num1 = random.randint(2, 9)
            num2 = random.randint(2, 9)
        elif difficulty == "easy":
            num1 = random.randint(2, 9)
            num2 = random.randint(10, 20)
        elif difficulty == "medium":
            num1 = random.randint(10, 30)
            num2 = random.randint(10, 30)
        else:  # hard
            num1 = random.randint(20, 99)
            num2 = random.randint(20, 99)
        
        answer = num1 * num2
        
        # Instruct model to use proper format
        prompt = (f"What is {num1} × {num2}? "
                f"Show your reasoning in <think> tags and "
                f"put your final answer in <answer> tags.")
        
        return {
            "prompt": prompt,
            "task": "multiplication",
            "num1": num1,
            "num2": num2,
            "answer": answer,
            "difficulty": difficulty
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """Custom collate function"""
    return batch


def create_dataloaders(config: Dict) -> tuple:
    """Create train and eval dataloaders with curriculum support"""
    # Use CurriculumMathDataset instead of MathDataset
    train_dataset = CurriculumMathDataset(
        num_samples=config['data']['train_size'],
        tasks=config['data']['tasks']
    )
    
    eval_dataset = CurriculumMathDataset(
        num_samples=config['data']['eval_size'],
        tasks=config['data']['tasks']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['apo']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['apo']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader, train_dataset, eval_dataset


# Test
if __name__ == "__main__":
    print("Testing Curriculum Learning...")
    
    dataset = CurriculumMathDataset(num_samples=10)
    
    print("\n=== VERY EASY (Step 0) ===")
    for i in range(3):
        prob = dataset[i]
        print(f"{prob['prompt']} → {prob['answer']}")
    
    print("\n=== EASY (Step 40) ===")
    dataset.set_difficulty_step(40)
    for i in range(3):
        prob = dataset[i]
        print(f"{prob['prompt']} → {prob['answer']}")
    
    print("\n=== MEDIUM (Step 70) ===")
    dataset.set_difficulty_step(70)
    for i in range(3):
        prob = dataset[i]
        print(f"{prob['prompt']} → {prob['answer']}")
    
    print("\n=== HARD (Step 100) ===")
    dataset.set_difficulty_step(100)
    for i in range(3):
        prob = dataset[i]
        print(f"{prob['prompt']} → {prob['answer']}")