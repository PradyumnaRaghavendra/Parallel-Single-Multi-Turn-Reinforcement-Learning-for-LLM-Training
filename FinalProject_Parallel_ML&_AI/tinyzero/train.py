"""
Complete training script for TinyZero
Includes all training loop logic, checkpointing, logging, and CURRICULUM LEARNING
"""
import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import time
import json

from tinyzero.models import PolicyModel, ReferenceModel
from tinyzero.data import create_dataloaders
from tinyzero.apo_trainer import APOTrainer
from tinyzero.evaluate import evaluate
from tinyzero.utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train TinyZero with A*PO')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (fewer steps)'
    )
    return parser.parse_args()


class Trainer:
    """Main trainer class that orchestrates training"""
    
    def __init__(self, config: dict, output_dir: str):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.output_dir / 'training.log'
        self.metrics_file = self.output_dir / 'metrics.json'
        
        # Initialize metrics storage
        self.metrics_history = {
            'train_loss': [],
            'train_reward': [],
            'eval_accuracy': [],
            'eval_accuracy_countdown': [],
            'eval_accuracy_multiplication': [],
            'steps': []
        }
        
        # Set seed
        set_seed(config.get('seed', 42))
        
        # Create dataloaders - NOW ALSO GETS DATASETS for curriculum
        print("Creating dataloaders with curriculum learning...")
        result = create_dataloaders(config)
        
        # Unpack based on what create_dataloaders returns
        if len(result) == 4:
            # New version with datasets
            self.train_loader, self.eval_loader, self.train_dataset, self.eval_dataset = result
        else:
            # Old version without datasets (backward compatible)
            self.train_loader, self.eval_loader = result
            self.train_dataset = None
            self.eval_dataset = None
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Eval samples: {len(self.eval_loader.dataset)}")
        
        # Initialize models
        print("Loading models...")
        self.policy = PolicyModel(
            config['model']['name'],
            device=config['model']['device']
        )
        self.ref_model = ReferenceModel(
            config['model']['ref_model'],
            device=config['model']['device']
        )
        print("Models loaded successfully!")
        
        # Initialize trainer
        self.apo_trainer = APOTrainer(
            self.policy,
            self.ref_model,
            config
        )
        
        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        
        # Meters for tracking
        self.loss_meter = AverageMeter()
        self.reward_meter = AverageMeter()
    
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with curriculum learning"""
        self.policy.train()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}"
        )
        
        for batch_idx, batch in pbar:
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                break
            
            # ========== CURRICULUM LEARNING UPDATE ==========
            # Update difficulty at curriculum milestones
            if (self.train_dataset is not None and 
                self.global_step > 0 and 
                self.global_step % 25 == 0):  # Update every 25 steps
                
                self.log(f"\n Updating curriculum difficulty at step {self.global_step}...")
                
                # Update difficulty in datasets
                self.train_dataset.set_difficulty_step(self.global_step)
                self.eval_dataset.set_difficulty_step(self.global_step)
                
                # Log current difficulty
                difficulty = self.train_dataset._get_difficulty_level(self.global_step)
                self.log(f"Curriculum level: {difficulty}")
            # ================================================
            
            # Training step
            try:
                loss, metrics = self.apo_trainer.train_step(batch)
                
                # Update meters
                self.loss_meter.update(loss)
                self.reward_meter.update(metrics['avg_reward'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{self.loss_meter.avg:.4f}",
                    'reward': f"{self.reward_meter.avg:.3f}",
                    'step': self.global_step
                })
                
            except Exception as e:
                self.log(f"Error in training step: {e}")
                continue
            
            # Logging
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.log(
                    f"Step {self.global_step}: "
                    f"Loss={self.loss_meter.avg:.4f}, "
                    f"Reward={self.reward_meter.avg:.3f}"
                )
                
                # Store metrics
                self.metrics_history['train_loss'].append(self.loss_meter.avg)
                self.metrics_history['train_reward'].append(self.reward_meter.avg)
                self.metrics_history['steps'].append(self.global_step)
                
                # Reset meters
                self.loss_meter.reset()
                self.reward_meter.reset()
            
            # Evaluation
            if self.global_step % self.config['training']['eval_every'] == 0:
                self.log(f"\n{'='*50}")
                self.log(f"Running evaluation at step {self.global_step}...")
                
                eval_metrics = evaluate(
                    self.policy,
                    self.eval_loader,
                    verbose=True
                )
                
                # Store eval metrics
                self.metrics_history['eval_accuracy'].append(eval_metrics['accuracy'])
                self.metrics_history['eval_accuracy_countdown'].append(
                    eval_metrics['accuracy_countdown']
                )
                self.metrics_history['eval_accuracy_multiplication'].append(
                    eval_metrics['accuracy_multiplication']
                )
                
                # Log results
                self.log(
                    f"Eval - Accuracy: {eval_metrics['accuracy']:.2%}, "
                    f"Countdown: {eval_metrics['accuracy_countdown']:.2%}, "
                    f"Multiplication: {eval_metrics['accuracy_multiplication']:.2%}"
                )
                
                # Save best model
                if eval_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_metrics['accuracy']
                    self.log(f"New best accuracy: {self.best_accuracy:.2%}")
                    save_checkpoint(
                        self.apo_trainer.policy.model,
                        self.apo_trainer.optimizer,
                        self.global_step,
                        self.output_dir / 'best_model.pt',
                        accuracy=self.best_accuracy
                    )
                
                self.log(f"{'='*50}\n")
                
                # Back to training mode
                self.policy.train()
            
            # Checkpointing
            if self.global_step % self.config['training']['save_every'] == 0:
                checkpoint_path = self.output_dir / f'checkpoint_{self.global_step}.pt'
                save_checkpoint(
                    self.apo_trainer.policy.model,
                    self.apo_trainer.optimizer,
                    self.global_step,
                    checkpoint_path
                )
                self.log(f"Saved checkpoint to {checkpoint_path}")
            
            # Save metrics periodically
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.save_metrics()
            
            self.global_step += 1
    
    def train(self):
        """Main training loop"""
        self.log("Starting training...")
        self.log(f"Config: {self.config}")
        
        # Log curriculum info if available
        if self.train_dataset is not None:
            self.log(" Curriculum Learning ENABLED")
            self.log("Difficulty will increase automatically during training")
        
        # Run initial evaluation
        self.log("Running initial evaluation...")
        initial_metrics = evaluate(self.policy, self.eval_loader, verbose=True)
        self.log(f"Initial accuracy: {initial_metrics['accuracy']:.2%}")
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.log(f"\n{'='*60}")
            self.log(f"Starting Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            self.log(f"{'='*60}")
            
            self.train_epoch(epoch)
            
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                self.log("Reached maximum training steps")
                break
        
        # Final evaluation
        self.log("\n" + "="*60)
        self.log("Running final evaluation...")
        final_metrics = evaluate(self.policy, self.eval_loader, verbose=True)
        
        self.log("\n" + "="*60)
        self.log("TRAINING COMPLETE!")
        self.log(f"Initial accuracy: {initial_metrics['accuracy']:.2%}")
        self.log(f"Final accuracy: {final_metrics['accuracy']:.2%}")
        self.log(f"Best accuracy: {self.best_accuracy:.2%}")
        self.log(f"Improvement: {(final_metrics['accuracy'] - initial_metrics['accuracy']):.2%}")
        self.log("="*60)
        
        # Save final model
        save_checkpoint(
            self.apo_trainer.policy.model,
            self.apo_trainer.optimizer,
            self.global_step,
            self.output_dir / 'final_model.pt',
            accuracy=final_metrics['accuracy']
        )
        
        # Save final metrics
        self.save_metrics()
        
        return final_metrics


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Debug mode - reduce steps
    if args.debug:
        print("Running in DEBUG mode - reducing steps")
        config['training']['max_steps'] = 50
        config['training']['eval_every'] = 20
        config['training']['save_every'] = 20
        config['data']['train_size'] = 100
        config['data']['eval_size'] = 20
    
    # Create trainer
    trainer = Trainer(config, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.global_step = load_checkpoint(
            trainer.apo_trainer.policy.model,
            trainer.apo_trainer.optimizer,
            args.resume
        )
        print(f"Resumed from step {trainer.global_step}")
    
    # Run evaluation only if specified
    if args.eval_only:
        print("Running evaluation only...")
        metrics = evaluate(trainer.policy, trainer.eval_loader, verbose=True)
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        return
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()