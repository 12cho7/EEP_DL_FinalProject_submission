# GitHub Upload Guide

## 🚀 Quick Upload Steps

### 1. Initialize Git (if not already initialized)

```bash
cd "/Users/morris/Documents/EEP 596D DL/Final_Predicting Human Preference Between LLM Responses with Fine-Tuning/EEP_DL_FinalProject_submission"

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: LLM Response Comparison Final Project"
```

### 2. Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `EEP_DL_FinalProject_submission`
3. Description: `Final project for EEP 596D - Predicting Human Preference Between LLM Responses with Fine-Tuning`
4. **Set to Public** (assignment requirement)
5. **Do NOT** check "Initialize with README" (you already have README.md)
6. Click "Create repository"

### 3. Connect and Upload

```bash
# Connect remote repository (replace YOUR_USERNAME if needed)
git remote add origin https://github.com/12cho7/EEP_DL_FinalProject_submission.git

# Set main branch
git branch -M main

# Upload
git push -u origin main
```

### 4. Verify

Go to your GitHub page to confirm:
https://github.com/12cho7/EEP_DL_FinalProject_submission

You should see all files there.

---

## 📋 Pre-upload Checklist

Make sure these files are included:

- [x] README.md
- [x] requirements.txt
- [x] src/main.py
- [x] src/utils.py
- [x] src/model.py
- [x] src/config.py
- [x] demo/demo.py
- [x] demo/demo.ipynb
- [x] demo/sample_inputs.json
- [x] .gitignore

---

## ⚠️ Important Reminders

### Do NOT Upload:

- Model files (too large, use Google Drive link)
- Data files (download from Kaggle)
- `__pycache__/` directories (in .gitignore)
- `.ipynb_checkpoints/` directories (in .gitignore)

### Remember to Update:

- README.md model download link (section 5) - **Already updated! ✅**

---

## 🔗 Submission URL

After uploading, your submission URL will be:
```
https://github.com/12cho7/EEP_DL_FinalProject_submission
```

Submit this URL to your professor.

---

## 🆘 Troubleshooting

### Authentication Issues

If push requires authentication:
1. Use Personal Access Token (PAT)
2. Or use GitHub CLI: `gh auth login`

### Files Too Large

If some files are too large:
- Check .gitignore is correct
- Ensure model files are not being uploaded

---

## ✅ After Completion

1. Confirm all files are on GitHub
2. Test links in README.md are correct
3. Submit GitHub URL to professor

Good luck! 🎉
