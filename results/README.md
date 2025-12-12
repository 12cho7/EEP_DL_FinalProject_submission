# Results Directory

This directory contains generated results from running the demo or making predictions.

## Expected Files

After running `demo/demo.py` or `demo/demo.ipynb`, you should see:

- `demo_predictions.csv` - Predictions from the demo script
- Other prediction files from inference runs

## File Format

The prediction CSV files have the following format:

```csv
sample_id,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
1,"What is machine learning?",...,...,0.6234,0.2145,0.1621
```

Where:
- `winner_model_a`: Probability that response A is better
- `winner_model_b`: Probability that response B is better  
- `winner_tie`: Probability that both responses are equally good
- Probabilities sum to ~1.0 for each row

