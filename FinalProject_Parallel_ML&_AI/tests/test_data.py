"""
Unit tests for data generation
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyzero.data import MathDataset


class TestMathDataset:
    """Test MathDataset class"""
    
    def test_dataset_length(self):
        """Test dataset has correct length"""
        dataset = MathDataset(num_samples=100)
        assert len(dataset) == 100
    
    def test_dataset_contains_both_tasks(self):
        """Test dataset contains both countdown and multiplication"""
        dataset = MathDataset(num_samples=100, tasks=['countdown', 'multiplication'])
        
        tasks = [item['task'] for item in dataset]
        
        assert 'countdown' in tasks
        assert 'multiplication' in tasks
    
    def test_dataset_only_countdown(self):
        """Test dataset with only countdown tasks"""
        dataset = MathDataset(num_samples=50, tasks=['countdown'])
        
        tasks = [item['task'] for item in dataset]
        
        assert all(task == 'countdown' for task in tasks)
    
    def test_multiplication_has_answer(self):
        """Test multiplication problems have correct answer"""
        dataset = MathDataset(num_samples=10, tasks=['multiplication'])
        
        for item in dataset:
            if item['task'] == 'multiplication':
                assert 'answer' in item
                assert item['answer'] == item['num1'] * item['num2']
    
    def test_countdown_has_target(self):
        """Test countdown problems have target"""
        dataset = MathDataset(num_samples=10, tasks=['countdown'])
        
        for item in dataset:
            if item['task'] == 'countdown':
                assert 'target' in item
                assert 100 <= item['target'] <= 999


class TestDataLoader:
    """Test dataloader creation"""
    
    def test_dataloader_creation(self):
        """Test that dataloaders are created correctly"""
        from tinyzero.data import create_dataloaders
        
        config = {
            'data': {
                'train_size': 100,
                'eval_size': 20,
                'tasks': ['countdown', 'multiplication']
            },
            'apo': {
                'batch_size': 4
            }
        }
        
        train_loader, eval_loader = create_dataloaders(config)
        
        assert len(train_loader.dataset) == 100
        assert len(eval_loader.dataset) == 20