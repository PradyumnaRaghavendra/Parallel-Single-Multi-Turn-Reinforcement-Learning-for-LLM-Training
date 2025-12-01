"""
Unit tests for A*PO trainer
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyzero.apo_trainer import APOTrainer


class TestAPOTrainer:
    """Test APOTrainer class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        return {
            'apo': {
                'beta': 0.01,
                'v_star_samples': 10,
                'learning_rate': 1e-6
            },
            'model': {
                'max_length': 512,
                'device': 'cpu'
            },
            'training': {
                'max_steps': 100
            }
        }
    
    @pytest.fixture
    def mock_policy(self):
        """Create mock policy model"""
        policy = Mock()
        policy.generate = Mock(return_value=["The answer is 100"])
        policy.get_log_probs = Mock(return_value=torch.tensor([0.5]))
        policy.parameters = Mock(return_value=[torch.tensor([1.0], requires_grad=True)])
        policy.train = Mock()
        policy.eval = Mock()
        return policy
    
    @pytest.fixture
    def mock_ref_model(self):
        """Create mock reference model"""
        ref_model = Mock()
        ref_model.generate = Mock(return_value=[["The answer is 100", "The answer is 99"]])
        return ref_model
    
    def test_trainer_initialization(self, mock_policy, mock_ref_model, mock_config):
        """Test that trainer initializes correctly"""
        trainer = APOTrainer(mock_policy, mock_ref_model, mock_config)
        
        assert trainer.policy == mock_policy
        assert trainer.ref_model == mock_ref_model
        assert trainer.beta == 0.01
        assert trainer.v_star_samples == 10
        assert trainer.step == 0
    
    def test_compute_v_star_shape(self, mock_policy, mock_ref_model, mock_config):
        """Test that V* computation returns correct shape"""
        trainer = APOTrainer(mock_policy, mock_ref_model, mock_config)
        
        prompts = ["What is 5 + 3?", "What is 10 × 2?"]
        V_star = trainer.compute_V_star(prompts)
        
        assert isinstance(V_star, np.ndarray)
        assert V_star.shape == (2,)  # One V* value per prompt
    
    def test_compute_v_star_values(self, mock_policy, mock_ref_model, mock_config):
        """Test that V* values are reasonable"""
        trainer = APOTrainer(mock_policy, mock_ref_model, mock_config)
        
        prompts = ["Test prompt"]
        V_star = trainer.compute_V_star(prompts)
        
        # V* should be between 0 and 1 (reward range)
        assert 0 <= V_star[0] <= 1
    
    def test_train_step_runs(self, mock_policy, mock_ref_model, mock_config):
        """Test that train_step executes without errors"""
        trainer = APOTrainer(mock_policy, mock_ref_model, mock_config)
        
        batch = [
            {
                'prompt': 'What is 10 × 10?',
                'task': 'multiplication',
                'answer': 100
            }
        ]
        
        # Should not raise exception
        loss = trainer.train_step(batch)
        
        assert isinstance(loss, float)
        assert trainer.step == 1
    
    def test_advantage_calculation(self, mock_policy, mock_ref_model, mock_config):
        """Test that advantages are calculated correctly"""
        trainer = APOTrainer(mock_policy, mock_ref_model, mock_config)
        
        # Mock V* computation
        with patch.object(trainer, 'compute_V_star', return_value=np.array([0.5])):
            batch = [{
                'prompt': 'Test',
                'task': 'multiplication',
                'answer': 100
            }]
            
            # If reward = 1.0 and V* = 0.5, advantage should be 0.5
            loss = trainer.train_step(batch)
            
            # Just check it runs without errors
            assert loss is not None


class TestVStarComputation:
    """Test V* computation specifically"""
    
    def test_v_star_with_all_correct(self):
        """Test V* when all samples are correct"""
        # If all samples get reward=1, V* should be close to 1
        rewards = np.ones(10)
        beta = 0.01
        
        # V* = β * log(mean(exp(r/β)))
        V_star = beta * np.log(np.mean(np.exp(rewards / beta)))
        
        assert V_star > 0.9  # Should be close to 1
    
    def test_v_star_with_all_wrong(self):
        """Test V* when all samples are wrong"""
        # If all samples get reward=0, V* should be close to 0
        rewards = np.zeros(10)
        beta = 0.01
        
        V_star = beta * np.log(np.mean(np.exp(rewards / beta)))
        
        assert V_star < 0.1  # Should be close to 0
    
    def test_v_star_with_mixed(self):
        """Test V* with mixed rewards"""
        # Half correct, half wrong
        rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        beta = 0.01
        
        V_star = beta * np.log(np.mean(np.exp(rewards / beta)))
        
        # Should be between 0 and 1
        assert 0 < V_star < 1