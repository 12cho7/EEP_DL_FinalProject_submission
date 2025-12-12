# Data Directory

This directory contains the dataset files for the LLM Response Comparison project.

## Data Source

The data is from the Kaggle competition:
[LLM Classification Fine-tuning](https://www.kaggle.com/competitions/llm-classification-finetuning)

## Required Files

- `train.csv` - Training data (57,477 samples)
- `test.csv` - Test data for predictions
- `sample_submission.csv` - Submission format example

## Download Instructions

### Method 1: Kaggle API (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (see main README.md)
# Download data
kaggle competitions download -c llm-classification-finetuning
unzip llm-classification-finetuning.zip -d data/
```

### Method 2: Manual Download

1. Go to [Kaggle Competition Page](https://www.kaggle.com/competitions/llm-classification-finetuning/data)
2. Download `train.csv` and `test.csv`
3. Place them in this directory

## Data Format

### train.csv
- `id`: Unique identifier
- `prompt`: The question/prompt text
- `response_a`: First LLM response
- `response_b`: Second LLM response
- `winner_model_a`: 1 if A wins, 0 otherwise
- `winner_model_b`: 1 if B wins, 0 otherwise
- `winner_tie`: 1 if tie, 0 otherwise

### test.csv
- `id`: Unique identifier
- `prompt`: The question/prompt text
- `response_a`: First LLM response
- `response_b`: Second LLM response

## Note

Due to file size, the actual data files are not included in the repository.
Please download them from Kaggle using the instructions above.

