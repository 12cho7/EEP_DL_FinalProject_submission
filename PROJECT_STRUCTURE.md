# Project Structure Explanation

## Overview

This project contains three main components that work together:

1. **`LLM_Response_Comparison_Colab.ipynb`** - Complete training notebook (in `Final_project/`)
2. **`src/`** - Modular Python code (reusable components)
3. **`demo/`** - Demo scripts for using the trained model

---

## Component Relationships

### 1. `LLM_Response_Comparison_Colab.ipynb` (Training Notebook)

**Location**: `Final_project/LLM_Response_Comparison_Colab.ipynb`

**Purpose**: 
- Complete, self-contained training pipeline
- Designed for Google Colab with GPU support
- Contains all code in one notebook for easy execution

**What it does**:
- Loads and preprocesses data
- Defines the model architecture
- Trains the model with LoRA fine-tuning
- Evaluates on validation set
- Makes predictions on test set
- Saves the trained model

**Key Features**:
- All code is in cells (no external imports from `src/`)
- Includes data augmentation
- Handles environment setup (Colab/Kaggle/Local)
- Complete training loop with early stopping

---

### 2. `src/` Directory (Modular Code)

**Purpose**: 
- Reusable, modular Python modules
- Can be imported by other scripts
- Separates concerns (data loading, model definition, training logic)

**Files**:

- **`main.py`** - Entry point for command-line usage
  - Can train models: `python src/main.py train`
  - Can make predictions: `python src/main.py predict`
  - Uses other modules from `src/`

- **`model.py`** - Model architecture definition
  - Defines the DeBERTa model with LoRA
  - Can be imported: `from src.model import create_model`

- **`utils.py`** - Helper functions
  - Text processing (parse_json, truncate)
  - Utility functions used across the project

- **`config.py`** - Hyperparameters and configuration
  - All training hyperparameters in one place
  - Model configuration (LoRA settings, etc.)
  - Easy to modify and experiment

**Relationship to Colab Notebook**:
- The Colab notebook contains **similar code** but is self-contained
- `src/` modules are **extracted versions** for reuse
- You can use `src/` modules in your own scripts
- The Colab notebook doesn't import from `src/` (it has everything inline)

---

### 3. `demo/` Directory (Demo Scripts)

**Purpose**: 
- Show how to **use** the trained model (not train it)
- Simple examples for loading and making predictions
- Demonstrates the model's functionality

**Files**:

- **`demo.py`** - Python script demo
  - Loads pre-trained model from `checkpoints/models_v3/`
  - Makes predictions on sample inputs
  - Saves results to `results/demo_predictions.csv`
  - **Does NOT train** - only inference

- **`demo.ipynb`** - Jupyter notebook version
  - Same functionality as `demo.py`
  - Interactive, can run in Colab or locally
  - Step-by-step cells showing model loading and prediction

- **`sample_inputs.json`** - Sample data for demo
  - Example prompts and responses
  - Used by demo scripts to show predictions

**Relationship to Colab Notebook**:
- Colab notebook: **Trains** the model
- Demo scripts: **Use** the trained model
- Demo assumes model is already trained and saved
- Demo is much simpler (just loading + inference)

---

## Workflow

### Training (Using Colab Notebook)

```
1. Upload LLM_Response_Comparison_Colab.ipynb to Colab
2. Upload train.csv and test.csv
3. Run all cells
4. Model is trained and saved to /content/models_v3/
5. Download models_v3.zip
```

### Using the Model (Using Demo)

```
1. Download models_v3.zip from Google Drive
2. Extract to checkpoints/models_v3/
3. Run: python demo/demo.py
4. See predictions in results/demo_predictions.csv
```

### Using Modular Code (Using src/)

```
1. Import modules: from src.model import create_model
2. Use in your own scripts
3. Customize training or inference
```

---

## Key Differences

| Component | Purpose | Training? | Inference? | Self-contained? |
|-----------|---------|-----------|-----------|-----------------|
| **Colab Notebook** | Complete training pipeline | ✅ Yes | ✅ Yes | ✅ Yes |
| **src/** | Reusable modules | ✅ Can train | ✅ Can infer | ❌ Needs imports |
| **demo/** | Show model usage | ❌ No | ✅ Yes | ✅ Yes (needs model) |

---

## Which One to Use?

- **Want to train a model?** → Use `LLM_Response_Comparison_Colab.ipynb`
- **Want to use a trained model?** → Use `demo/demo.py` or `demo/demo.ipynb`
- **Want to build custom scripts?** → Import from `src/`
- **Submitting for assignment?** → Include all three (shows complete project)

---

## File Organization

```
EEP_DL_FinalProject_submission/
├── README.md                    # Project documentation
├── requirements.txt            # Dependencies
├── src/                        # Modular code
│   ├── main.py                 # Entry point
│   ├── model.py                # Model definition
│   ├── utils.py                # Helper functions
│   └── config.py               # Hyperparameters
├── demo/                       # Demo scripts
│   ├── demo.py                 # Python demo
│   ├── demo.ipynb              # Notebook demo
│   └── sample_inputs.json      # Sample data
├── data/                       # Data directory (README explains download)
├── checkpoints/                # Model directory (README explains download)
└── results/                    # Output directory

Final_project/                  # (Not in submission)
└── LLM_Response_Comparison_Colab.ipynb  # Complete training notebook
```

---

## Summary

- **Colab Notebook** = Everything in one place for training
- **src/** = Modular code for reuse
- **demo/** = Simple examples for using trained model

All three serve different purposes and complement each other!

