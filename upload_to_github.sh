#!/bin/bash

# GitHub Upload Script
# Usage: ./upload_to_github.sh

echo "=========================================="
echo "GitHub Upload Script"
echo "=========================================="

# Check if in correct directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script in EEP_DL_FinalProject_submission directory"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git..."
    git init
fi

# Add all files
echo ""
echo "Adding files to Git..."
git add .

# Check status
echo ""
echo "File status:"
git status --short

# First commit
echo ""
echo "Creating commit..."
git commit -m "Initial commit: LLM Response Comparison Final Project"

echo ""
echo "=========================================="
echo "✓ Local Git ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/new to create new repository"
echo "2. Repository name: EEP_DL_FinalProject_submission"
echo "3. Set to Public"
echo "4. Do NOT initialize with README"
echo "5. After creating, run these commands:"
echo ""
echo "   git remote add origin https://github.com/12cho7/EEP_DL_FinalProject_submission.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "=========================================="
