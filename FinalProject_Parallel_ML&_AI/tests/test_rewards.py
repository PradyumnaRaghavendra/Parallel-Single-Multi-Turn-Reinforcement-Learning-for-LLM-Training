"""
Unit tests for reward computation
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyzero.rewards import compute_reward, extract_answer


class TestExtractAnswer:
    """Test answer extraction"""
    
    def test_extract_from_answer_is(self):
        """Test extracting from 'the answer is X'"""
        text = "Let me calculate. The answer is 100."
        answer = extract_answer(text)
        assert answer == "100"
    
    def test_extract_from_equals(self):
        """Test extracting from '= X'"""
        text = "5 × 20 = 100"
        answer = extract_answer(text)
        assert answer == "100"
    
    def test_extract_last_number(self):
        """Test fallback to last number"""
        text = "First I got 50, then 75, finally 100"
        answer = extract_answer(text)
        assert answer == "100"
    
    def test_extract_with_commas(self):
        """Test extracting numbers with commas"""
        text = "The answer is 1,234,567"
        answer = extract_answer(text)
        assert answer == "1234567"  


class TestComputeReward:
    """Test reward computation"""
    
    def test_correct_multiplication(self):
        """Test reward for correct multiplication"""
        problem = {
            'task': 'multiplication',
            'answer': 100,
            'prompt': 'What is 10 × 10?'
        }
        
        text = "The answer is 100"
        reward = compute_reward(text, problem)
        
        assert reward == 1.0
    
    def test_incorrect_multiplication(self):
        """Test reward for incorrect multiplication"""
        problem = {
            'task': 'multiplication',
            'answer': 100,
            'prompt': 'What is 10 × 10?'
        }
        
        text = "The answer is 99"
        reward = compute_reward(text, problem)
        
        assert reward == 0.0
    
    def test_correct_countdown(self):
        """Test reward for correct countdown"""
        problem = {
            'task': 'countdown',
            'target': 857,
            'prompt': 'Make 857'
        }
        
        text = "Using these operations, I get 857"
        reward = compute_reward(text, problem)
        
        assert reward == 1.0
    
    def test_no_answer_found(self):
        """Test reward when no answer is found"""
        problem = {
            'task': 'multiplication',
            'answer': 100,
            'prompt': 'Test'
        }
        
        text = "I don't know"
        reward = compute_reward(text, problem)
        
        assert reward == 0.0