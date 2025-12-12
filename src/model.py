"""
Model definitions for LLM classification fine-tuning.
"""
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
from typing import Optional, Dict, Any


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    use_peft: bool = True,
    peft_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda"
):
    """
    Load model and tokenizer for fine-tuning.
    
    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of classification labels
        use_peft: Whether to use PEFT (LoRA)
        peft_config: Configuration for PEFT
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    # Apply PEFT if enabled
    if use_peft and peft_config:
        # Determine task type based on model architecture
        task_type = TaskType.SEQ_CLS
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=task_type,
            r=peft_config.get('r', 16),
            lora_alpha=peft_config.get('lora_alpha', 32),
            lora_dropout=peft_config.get('lora_dropout', 0.1),
            target_modules=peft_config.get('target_modules', ['query', 'value']),
            bias="none"
        )
        
        # Apply PEFT
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    return model, tokenizer


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metrics
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

