# Predicting Human Preference Between LLM Responses with Fine-Tuning

**Author**: Morris Chou  
**Course**: EEP 596D Deep Learning (Fall 2025)  
**Competition**: [Kaggle - LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning)

---

## 1. Project Overview

This project fine-tunes a DeBERTa-v3-base model to predict human preferences between two LLM responses. Given a prompt and two responses (A and B), the model predicts which response is better, or if they are equally good (tie).

### Key Features

- **Model**: DeBERTa-v3-base with LoRA fine-tuning
- **Task**: Multi-class classification (A wins, B wins, Tie)
- **Evaluation Metric**: Log Loss
- **Training Data**: 57,477 samples (with augmentation: 114,954)
- **Best Validation Log Loss**: 1.0735
- **Best Validation Accuracy**: 39.78%

### Technical Approach

- **Base Model**: `microsoft/deberta-v3-base`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) - Parameter-efficient fine-tuning
- **Data Augmentation**: Swap A/B responses to prevent position bias
- **Training Strategy**: Cosine learning rate scheduler, early stopping, label smoothing

---

## 2. Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~10GB disk space for model and data

### Installation

1. **Clone the repository** (or download the project files)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**:
   - Option A: Download from [Kaggle Competition](https://www.kaggle.com/competitions/llm-classification-finetuning/data)
   - Option B: Use Kaggle API:
     ```bash
     kaggle competitions download -c llm-classification-finetuning
     unzip llm-classification-finetuning.zip -d data/
     ```

4. **Download pre-trained model** (see [Pre-trained Model Link](#4-pre-trained-model-link) section below)

---

## 3. How to Run

### Option 1: Run Demo Script (Recommended)

The easiest way to test the model is using the demo script:

```bash
python demo/demo.py
```

This will:
- Load the pre-trained model from `checkpoints/models_v3/`
- Run predictions on sample inputs
- Save results to `results/demo_predictions.csv`

**Note**: Make sure you have downloaded the model (see section 5) and extracted it to `checkpoints/models_v3/`

### Option 2: Run Demo Notebook

Alternatively, use the Jupyter notebook:

```bash
jupyter notebook demo/demo.ipynb
```

Or in Google Colab:
1. Upload `demo/demo.ipynb` to Colab
2. Update `MODEL_PATH` to point to your model location
3. Run all cells

### Option 3: Run Training Notebook

For full training (requires GPU, ~4-6 hours on T4):

```bash
# Upload to Google Colab
# 1. Enable GPU: Runtime → Change runtime type → GPU → T4
# 2. Upload train.csv and test.csv
# 3. Run all cells in LLM_Response_Comparison_Colab.ipynb
```

### Option 3: Use Command Line Scripts

**Train model**:
```bash
python src/main.py train --config config.yaml
```

**Make predictions**:
```bash
python src/main.py predict \
    --model_path checkpoints/models_v3 \
    --test_data data/test.csv \
    --output outputs/predictions.csv
```

### Quick Demo Script

```python
# demo.py
from demo.demo_utils import load_model, predict_single

# Load model
model, tokenizer = load_model('checkpoints/models_v3')

# Make prediction
result = predict_single(
    prompt="What is machine learning?",
    response_a="Machine learning is a subset of AI...",
    response_b="ML is when computers learn from data.",
    model=model,
    tokenizer=tokenizer
)

print(f"A wins: {result['winner_model_a']:.4f}")
print(f"B wins: {result['winner_model_b']:.4f}")
print(f"Tie: {result['winner_tie']:.4f}")
```

---

## 4. Expected Output

### Demo Notebook Output

When running `demo/demo.ipynb`, you should see:

```
Loading tokenizer...
✓ Tokenizer loaded
Loading model...
✓ Model loaded successfully
Device: cuda

Prediction Results:
============================================================
Prompt: What is machine learning?

Response A: Machine learning is a subset of artificial intelligence...
Response B: Machine learning is when computers learn stuff from data.

Probabilities:
  A wins: 0.6234 (62.34%)
  B wins: 0.2145 (21.45%)
  Tie:    0.1621 (16.21%)

✓ Predicted winner: A
============================================================
```

### Training Output

When training completes, you should see:

```
Training completed!
Final training loss: 1.0834

Validation Results:
  Log Loss: 1.0735
  Accuracy: 0.3978
  Acc (A): 0.2792
  Acc (B): 0.5053
  Acc (Tie): 0.4102

✓ Model saved to: /content/models_v3
```

### Submission File

The final `submission.csv` should have the format:

```csv
id,winner_model_a,winner_model_b,winner_tie
136060,0.263793,0.256553,0.479654
211333,0.360080,0.367592,0.272328
1233961,0.294552,0.286957,0.418490
```

Where probabilities sum to ~1.0 for each row.

---

## 5. Pre-trained Model Link

### Model Download

The trained model (v3.0) is available at:

**⚠️ IMPORTANT**: Please upload your `models_v3` folder to a file-sharing service and update the link below.

- **Google Drive**: [Download Link](https://drive.google.com/file/d/1yhMGtBz2yAPdTEPimi_T9wBZHKsVKt8o/view?usp=sharing)
- **Hugging Face Hub**: [Model Card](https://huggingface.co/YOUR_USERNAME/llm-response-comparison-v3) - *Optional*

### Quick Upload Guide

**In Colab** (after training completes):

```python
# Upload to Google Drive
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.make_archive('models_v3', 'zip', '/content/models_v3')
shutil.copy('models_v3.zip', '/content/drive/MyDrive/models_v3.zip')

# Then share the file and get the link
```

See [MODEL_UPLOAD_GUIDE.md](MODEL_UPLOAD_GUIDE.md) for detailed instructions.

### Model Details

- **Base Model**: `microsoft/deberta-v3-base`
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.1)
- **Model Size**: ~700MB (compressed: ~200-300MB)
- **Training**: 5 epochs, 15,000 steps (92.7% of planned 16,170 steps)
- **Performance**: Validation Log Loss: 1.0735, Accuracy: 39.78%

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel

# Download and extract models_v3.zip to checkpoints/models_v3/
MODEL_PATH = 'checkpoints/models_v3'

# Load
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', config=config
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()
```

See `demo/demo.ipynb` for a complete example.

---

## 6. Project Structure

```
Final_project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration file
├── src/                           # Source code
│   ├── main.py                    # Entry point
│   ├── model.py                   # Model definition
│   ├── data_loader.py             # Data loading
│   ├── trainer.py                 # Training logic
│   ├── utils.py                   # Helper functions
│   └── config.py                  # Hyperparameters (detailed)
├── demo/                          # Demo code
│   └── demo.ipynb                 # Demo notebook (run this!)
├── data/                          # Data directory
│   ├── train.csv                  # Training data (download from Kaggle)
│   ├── test.csv                   # Test data (download from Kaggle)
│   └── sample_submission.csv      # Submission format
├── checkpoints/                   # Model checkpoints
│   └── models_v3/                 # Trained model (download from link above)
├── outputs/                       # Output results
│   └── test_submission.csv        # Predictions
├── LLM_Response_Comparison_Colab.ipynb  # Full training notebook (Colab)
└── LLM_Response_Comparison_v3.ipynb      # Training notebook (Kaggle/Local)
```

---

## 7. Hyperparameters and Training Setup

All hyperparameters are documented in `src/config.py`. Key settings:

### Model Configuration
- **Model**: `microsoft/deberta-v3-base`
- **Max Length**: 1280 tokens
- **LoRA**: r=16, alpha=32, dropout=0.1

### Training Configuration
- **Epochs**: 5
- **Batch Size**: 2 × 16 (gradient accumulation) = 32 effective
- **Learning Rate**: 2e-5
- **Scheduler**: Cosine decay
- **Early Stopping**: Patience=3
- **Label Smoothing**: 0.1

### Data Configuration
- **Augmentation**: Swap A/B responses (doubles training data)
- **Validation Split**: 10% (stratified)
- **Smart Truncation**: Keep head and tail of long texts

See `src/config.py` for complete hyperparameter explanations.

---

## 8. Reproducibility

### Seed
All random operations use `seed=42` for reproducibility.

### Environment
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- PEFT 0.6+

### Training Logs
Training was performed on:
- **Platform**: Google Colab
- **GPU**: NVIDIA A100-SXM4-40GB
- **Training Time**: ~10.5 hours (15,000 steps)
- **Final Checkpoint**: Step 15,000 (best model loaded)

### Reproducing Results

1. **Download data** from Kaggle competition
2. **Download model** from the link in section 5
3. **Run demo**: `jupyter notebook demo/demo.ipynb`
4. **Or retrain**: Follow instructions in `LLM_Response_Comparison_Colab.ipynb`

All code includes detailed comments explaining the implementation.

---

## 9. Acknowledgments

### Data Source
- **Competition**: [Kaggle - LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
- Data provided by competition organizers

### Libraries and Tools
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model framework
- [PEFT](https://huggingface.co/docs/peft) - Parameter-efficient fine-tuning
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Google Colab](https://colab.research.google.com/) - Training platform

### References
- DeBERTa: [He et al., 2021](https://arxiv.org/abs/2006.03654)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- See [REFERENCES.md](REFERENCES.md) for complete reference list

---

## 10. License

This project is for educational purposes only.

---

## Contact

For questions or issues, please refer to the project repository or contact the author.

**Note**: Remember to update the pre-trained model link in section 5 after uploading your model!
