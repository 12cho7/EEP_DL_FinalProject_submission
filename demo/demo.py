"""
Demo script for LLM Response Comparison Model
This script demonstrates how to load the trained model and make predictions.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')


# Configuration
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LENGTH = 1280
NUM_LABELS = 3

# Model path - Update this to your model location
# Option 1: Local path (if downloaded)
MODEL_PATH = os.path.join('checkpoints', 'models_v3')

# Option 2: Google Drive (if using Colab)
# MODEL_PATH = '/content/drive/MyDrive/models_v3'

# Option 3: Hugging Face Hub (if uploaded)
# MODEL_PATH = 'YOUR_USERNAME/llm-response-comparison-v3'


def parse_json(text):
    """Parse JSON text safely"""
    try:
        p = json.loads(text) if isinstance(text, str) else text
        return '\n'.join([str(i) for i in p]) if isinstance(p, list) else str(p)
    except:
        return str(text)


def truncate(text, max_chars):
    """Smart truncation: keep head and tail"""
    if len(text) <= max_chars:
        return text
    h = int(max_chars * 0.6)
    t = max_chars - h - 10
    return text[:h] + "\n[...]\n" + text[-t:]


def load_model(model_path=MODEL_PATH):
    """Load the trained model and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("✓ Model loaded successfully")
    print(f"Device: {device}")
    
    return model, tokenizer


def predict_single(prompt, response_a, response_b, model, tokenizer, max_length=MAX_LENGTH):
    """
    Predict which response is better for a single example.
    
    Args:
        prompt: The question/prompt text
        response_a: First response
        response_b: Second response
        model: Loaded model
        tokenizer: Loaded tokenizer
        max_length: Maximum sequence length
    
    Returns:
        dict with probabilities for A wins, B wins, and tie
    """
    max_chars = (max_length * 4) // 3
    
    prompt_text = truncate(parse_json(prompt), max_chars // 4)
    resp_a = truncate(parse_json(response_a), max_chars * 3 // 8)
    resp_b = truncate(parse_json(response_b), max_chars * 3 // 8)
    
    text = f"Compare responses:\n\nQ: {prompt_text}\n\n[A]: {resp_a}\n\n[B]: {resp_b}\n\nBetter?"
    
    enc = tokenizer(text, truncation=True, padding='max_length', 
                    max_length=max_length, return_tensors='pt')
    
    input_ids = enc['input_ids'].to(model.device)
    attention_mask = enc['attention_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    probs = probs / probs.sum()
    
    return {
        'winner_model_a': float(probs[0]),
        'winner_model_b': float(probs[1]),
        'winner_tie': float(probs[2])
    }


def run_demo():
    """Run the demo with sample inputs"""
    print("=" * 60)
    print("LLM Response Comparison - Demo")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("\nPlease:")
        print("1. Download the model from the link in README.md")
        print("2. Extract to checkpoints/models_v3/")
        print("3. Or update MODEL_PATH in this script")
        return
    
    # Load model
    model, tokenizer = load_model()
    
    # Load sample inputs from demo folder
    print("\n" + "=" * 60)
    print("Running predictions on sample inputs...")
    print("=" * 60)
    
    # Try to load from sample_inputs.json, fallback to hardcoded samples
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(demo_dir, 'sample_inputs.json')
    
    if os.path.exists(sample_file):
        print(f"Loading samples from {sample_file}...")
        with open(sample_file, 'r') as f:
            samples = json.load(f)
        print(f"✓ Loaded {len(samples)} samples from file")
    else:
        print("⚠ Sample file not found, using hardcoded samples")
        samples = [
            {
                'prompt': "What is machine learning?",
                'response_a': "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves algorithms that can identify patterns and make decisions based on data.",
                'response_b': "Machine learning is when computers learn stuff from data."
            },
            {
                'prompt': "Explain neural networks",
                'response_a': "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.",
                'response_b': "Neural networks are a type of machine learning model that uses layers of nodes to process information, similar to how the human brain works."
            }
        ]
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {sample['prompt']}")
        print(f"Response A: {sample['response_a'][:80]}...")
        print(f"Response B: {sample['response_b'][:80]}...")
        
        result = predict_single(
            sample['prompt'],
            sample['response_a'],
            sample['response_b'],
            model, tokenizer, MAX_LENGTH
        )
        
        results.append({
            'sample_id': i,
            'prompt': sample['prompt'],
            'response_a': sample['response_a'],
            'response_b': sample['response_b'],
            **result
        })
        
        print(f"\nPredictions:")
        print(f"  A wins: {result['winner_model_a']:.4f} ({result['winner_model_a']*100:.2f}%)")
        print(f"  B wins: {result['winner_model_b']:.4f} ({result['winner_model_b']*100:.2f}%)")
        print(f"  Tie:    {result['winner_tie']:.4f} ({result['winner_tie']*100:.2f}%)")
        
        winner = 'A' if result['winner_model_a'] > result['winner_model_b'] and result['winner_model_a'] > result['winner_tie'] else 'B' if result['winner_model_b'] > result['winner_tie'] else 'Tie'
        print(f"  → Predicted winner: {winner}")
    
    # Save results to results/ folder
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results)
    output_path = 'results/demo_predictions.csv'
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print(f"✓ Results saved to: {output_path}")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    run_demo()

