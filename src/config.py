"""
Configuration file for hyperparameters and training setup.
This file documents all hyperparameters used in the project.
"""

# Model Configuration
MODEL_CONFIG = {
    'name': 'microsoft/deberta-v3-base',  # Base model: DeBERTa-v3-base
    'max_length': 1280,  # Maximum sequence length (tokens)
    'num_labels': 3,  # Number of classes: A wins, B wins, Tie
    'use_lora': True,  # Use LoRA for parameter-efficient fine-tuning
    'lora_config': {
        'r': 16,  # LoRA rank
        'alpha': 32,  # LoRA alpha (scaling factor)
        'dropout': 0.1,  # LoRA dropout
        'target_modules': ['query_proj', 'value_proj', 'key_proj'],  # Modules to apply LoRA
        'bias': 'none',  # Bias handling
        'modules_to_save': ['classifier']  # Modules to save (not use LoRA)
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 5,  # Number of training epochs
    'batch_size': 2,  # Batch size per device
    'gradient_accumulation_steps': 16,  # Effective batch size = batch_size * gradient_accumulation_steps = 32
    'learning_rate': 2e-5,  # Learning rate
    'weight_decay': 0.01,  # Weight decay for regularization
    'warmup_ratio': 0.1,  # Warmup steps ratio (10% of total steps)
    'validation_split': 0.1,  # Validation set ratio (10%)
    'early_stopping_patience': 3,  # Early stopping patience (epochs)
    'seed': 42,  # Random seed for reproducibility
    'fp16': True,  # Use mixed precision training (FP16)
    'label_smoothing_factor': 0.1,  # Label smoothing for regularization
    'lr_scheduler_type': 'cosine',  # Learning rate scheduler: cosine decay
    'save_strategy': 'steps',  # Save checkpoint strategy
    'eval_strategy': 'steps',  # Evaluation strategy
    'save_total_limit': 2,  # Maximum number of checkpoints to keep
    'load_best_model_at_end': True,  # Load best model at end of training
    'metric_for_best_model': 'log_loss',  # Metric for best model selection
    'greater_is_better': False,  # Lower log_loss is better
}

# Data Configuration
DATA_CONFIG = {
    'use_augmentation': True,  # Use data augmentation (swap A/B responses)
    'augmentation_method': 'swap_ab',  # Augmentation method
    'stratify_split': True,  # Use stratified split for train/val
    'smart_truncation': True,  # Use smart truncation (keep head and tail)
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'use_cuda': True,  # Use CUDA if available
    'dataloader_num_workers': 0,  # Number of data loading workers (0 for Windows/Mac compatibility)
    'pin_memory': True,  # Pin memory for faster data transfer
}

# Output Configuration
OUTPUT_CONFIG = {
    'output_dir': 'checkpoints/models_v3',  # Output directory for checkpoints
    'logging_dir': 'logs',  # Directory for training logs
    'results_dir': 'results',  # Directory for results
}

# Evaluation Configuration
EVAL_CONFIG = {
    'metrics': ['log_loss', 'accuracy'],  # Metrics to compute
    'compute_per_class_accuracy': True,  # Compute per-class accuracy
}

# Version Information
VERSION = '3.0'
DESCRIPTION = """
LLM Response Comparison - v3.0

Improvements over v2.0:
- Larger model: DeBERTa-v3-base (vs small)
- Longer context: 1280 tokens (vs 1024)
- Data augmentation: Enabled (swap A/B to prevent position bias)
- More epochs: 5 (vs 3)
- Higher learning rate: 2e-5 (vs 1e-5)
- Better regularization: Label smoothing + weight decay
"""

# Hyperparameter Explanation
HYPERPARAMETER_EXPLANATION = """
Hyperparameter Selection Rationale:

1. Model: DeBERTa-v3-base
   - Reason: Better performance than small variant, good balance between speed and accuracy
   - Alternative considered: DeBERTa-v3-small (faster but less accurate)

2. Max Length: 1280 tokens
   - Reason: Longer context allows capturing more information from responses
   - Trade-off: Longer sequences require more memory and computation

3. LoRA Configuration:
   - r=16: Good balance between parameter efficiency and model capacity
   - alpha=32: Standard scaling (alpha = 2 * r)
   - dropout=0.1: Prevents overfitting

4. Batch Size: 2 with gradient accumulation 16 (effective batch size = 32)
   - Reason: GPU memory constraints, gradient accumulation simulates larger batch
   - Effective batch size chosen based on common practices for fine-tuning

5. Learning Rate: 2e-5
   - Reason: Standard learning rate for transformer fine-tuning
   - Higher than v2 (1e-5) for faster convergence with more epochs

6. Epochs: 5
   - Reason: More epochs allow model to learn better patterns
   - Early stopping prevents overfitting

7. Learning Rate Scheduler: Cosine
   - Reason: Smooth decay helps model converge better than linear decay

8. Data Augmentation: Swap A/B
   - Reason: Prevents position bias (model always preferring A or B)
   - Doubles training data effectively

9. Label Smoothing: 0.1
   - Reason: Prevents overconfidence, improves generalization

10. Warmup Ratio: 0.1
    - Reason: Gradual learning rate increase at start stabilizes training
"""

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Version: {VERSION}")
    print(DESCRIPTION)
    print("\n" + "=" * 60)
    print("Model Configuration:")
    print(MODEL_CONFIG)
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print(TRAINING_CONFIG)
    print("\n" + "=" * 60)
    print(HYPERPARAMETER_EXPLANATION)

