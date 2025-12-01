"""
Setup file for TinyZero + RAGEN package
Supports both single-turn (TinyZero) and multi-turn (RAGEN) agent training
"""
from setuptools import setup, find_packages

setup(
    name="tinyzero-ragen",
    version="0.2.0",
    description="Efficient single-turn and multi-turn RL agent training with A*PO",
    author="Your Name",
    packages=find_packages(),  # Finds BOTH tinyzero/ and ragen/ automatically
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",  # Added for faster training
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "webshop": [
            # Future: Add WebShop-specific dependencies here
            # "webshop>=0.1.0",  (when you integrate real WebShop)
        ],
    },
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'tinyzero-train=tinyzero.train:main',
            'ragen-train=ragen.train_ragen:main',
        ],
    },
)