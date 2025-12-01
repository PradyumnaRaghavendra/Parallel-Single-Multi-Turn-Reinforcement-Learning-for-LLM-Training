"""
V* Caching System
Saves compute by reusing V* calculations for identical prompts.
Works for BOTH multiplication and countdown tasks.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional

class VStarCache:
    """Smart cache for V* computations"""
    
    def __init__(self, cache_dir: str = "./vstar_cache"):  # ← Changed default!
        """
        Initialize V* cache
        
        Args:
            cache_dir: Directory to store cache files
                      Default: "./vstar_cache" (local directory)
                      On Modal: "/vstar_cache" (mounted volume)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.enabled = True
        print(f"V* cache initialized at {cache_dir}")
    
    def _get_cache_key(self, prompt: str, config: dict) -> str:
        """Generate unique key for prompt + config combination"""
        key_data = {
            'prompt': prompt,
            'v_star_samples': config['apo']['v_star_samples'],
            'temperature': config['sampling'].get('temperature', 1.0),
            'model': config['model']['ref_model']
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, config: dict) -> Optional[Dict]:
        """Try to load cached V* data"""
        if not self.enabled:
            return None
        
        cache_key = self._get_cache_key(prompt, config)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.hits += 1
                    return data
            except Exception as e:
                print(f"  Cache read error: {e}")
        
        self.misses += 1
        return None
    
    def save(self, prompt: str, config: dict, vstar_data: Dict):
        """Save V* data to cache"""
        if not self.enabled:
            return
        
        try:
            cache_key = self._get_cache_key(prompt, config)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(vstar_data, f)
        except Exception as e:
            print(f"  Cache write error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total': total
        }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        if stats['total'] > 0:
            print(f"\nV* Cache: {stats['hits']}/{stats['total']} hits ({stats['hit_rate']*100:.1f}% - saved compute!)")