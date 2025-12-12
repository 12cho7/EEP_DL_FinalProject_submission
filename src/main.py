"""
Main entry point for LLM Response Comparison project.
This script provides a simple interface to train and evaluate the model.
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, set_seed, get_device
from src.data_loader import load_data, create_datasets
from src.model import load_model_and_tokenizer
from src.trainer import create_training_arguments, train_model
import pandas as pd
import torch


def train(config_path: str = "config.yaml", resume_from: str = None):
    """
    Train the model.
    
    Args:
        config_path: Path to configuration file
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("=" * 60)
    print("LLM Response Comparison - Training")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Get device
    device = get_device(config['hardware']['use_cuda'])
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(config['data']['train_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset = create_datasets(
        train_df,
        tokenizer=None,  # Will be created in model loading
        max_length=config['model']['max_length'],
        validation_split=config['training']['validation_split']
    )
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        use_peft=config['model']['use_lora'],
        peft_config=config['model'].get('lora_config', {}),
        device=str(device)
    )
    
    # Recreate datasets with tokenizer
    train_dataset, val_dataset = create_datasets(
        train_df,
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        validation_split=config['training']['validation_split']
    )
    
    # Create training arguments
    print("\nSetting up training...")
    training_args = create_training_arguments(
        output_dir=config['training']['output_dir'],
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        fp16=config['training']['fp16'],
        seed=config['training']['seed']
    )
    
    # Train model
    print("\nStarting training...")
    trainer = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        resume_from_checkpoint=resume_from
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config['training']['output_dir']}")
    print("=" * 60)


def predict(model_path: str, test_data_path: str, output_path: str):
    """
    Make predictions on test data.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test CSV file
        output_path: Path to save predictions
    """
    print("=" * 60)
    print("LLM Response Comparison - Prediction")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    print(f"Test samples: {len(test_df)}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_path,
        num_labels=3,
        use_peft=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = create_datasets(
        test_df,
        tokenizer=tokenizer,
        max_length=512,
        validation_split=0.0,
        is_test=True
    )
    
    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for item in test_dataset:
            input_ids = item['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(probs)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': [p[0] for p in predictions],
        'winner_model_b': [p[1] for p in predictions],
        'winner_tie': [p[2] for p in predictions]
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Response Comparison - Final Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python src/main.py train --config config.yaml
  
  # Train with resume
  python src/main.py train --config config.yaml --resume checkpoints/checkpoint-1000
  
  # Make predictions
  python src/main.py predict --model_path checkpoints/best_model --test_data data/test.csv --output results/predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (optional)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    predict_parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save predictions CSV'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(config_path=args.config, resume_from=args.resume)
    elif args.command == 'predict':
        predict(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_path=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

