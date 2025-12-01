"""
Test script to verify all components work
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from tinyzero import models, data, rewards, utils, apo_trainer
        from tinyzero import train, evaluate, visualization
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generation():
    """Test data generation"""
    print("\nTesting data generation...")
    try:
        from tinyzero.data import MathDataset
        
        dataset = MathDataset(num_samples=10)
        assert len(dataset) == 10
        
        problem = dataset[0]
        assert 'prompt' in problem
        assert 'task' in problem
        
        print(f"✓ Data generation works!")
        print(f"  Sample problem: {problem['prompt'][:50]}...")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False

def test_reward_computation():
    """Test reward function"""
    print("\nTesting reward computation...")
    try:
        from tinyzero.rewards import compute_reward, extract_answer
        
        # Test multiplication
        problem = {
            'task': 'multiplication',
            'answer': 100,
            'prompt': 'Test'
        }
        
        correct_text = "The answer is 100"
        wrong_text = "The answer is 99"
        
        reward1 = compute_reward(correct_text, problem)
        reward2 = compute_reward(wrong_text, problem)
        
        assert reward1 == 1.0, "Should reward correct answer"
        assert reward2 == 0.0, "Should not reward wrong answer"
        
        print("✓ Reward computation works!")
        return True
    except Exception as e:
        print(f"✗ Reward computation failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting config loading...")
    try:
        from tinyzero.utils import load_config
        
        config = load_config('configs/default.yaml')
        assert 'model' in config
        assert 'apo' in config
        assert 'training' in config
        
        print("✓ Config loading works!")
        print(f"  Model: {config['model']['name']}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_reward_computation,
        test_config_loading
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("\n✓ ALL TESTS PASSED! Ready to train!")
    else:
        print("\n✗ Some tests failed. Fix errors before training.")
    
    return all(results)

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)