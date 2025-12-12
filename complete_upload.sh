#!/bin/bash

# Complete Upload Script
# This script helps you complete the final step of uploading to GitHub

echo "=========================================="
echo "GitHub Upload - Final Step"
echo "=========================================="
echo ""

# Check if repository exists
echo "Checking if repository exists..."
if git ls-remote --exit-code origin main &>/dev/null; then
    echo "✓ Repository exists!"
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin main
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ SUCCESS! Upload completed!"
        echo "=========================================="
        echo ""
        echo "Your repository is now available at:"
        echo "https://github.com/12cho7/EEP_DL_FinalProject_submission"
        echo ""
        echo "Verifying upload..."
        sleep 2
        open "https://github.com/12cho7/EEP_DL_FinalProject_submission" 2>/dev/null || echo "Please visit: https://github.com/12cho7/EEP_DL_FinalProject_submission"
        exit 0
    else
        echo "❌ Push failed. Please check your authentication."
        exit 1
    fi
else
    echo "⚠️  Repository not found on GitHub yet."
    echo ""
    echo "=========================================="
    echo "Step 1: Create Repository on GitHub"
    echo "=========================================="
    echo ""
    echo "Please follow these steps:"
    echo ""
    echo "1. Open your browser and go to:"
    echo "   https://github.com/new"
    echo ""
    echo "2. Fill in the form:"
    echo "   - Repository name: EEP_DL_FinalProject_submission"
    echo "   - Description: Final project for EEP 596D - Predicting Human Preference Between LLM Responses with Fine-Tuning"
    echo "   - Visibility: PUBLIC (required for assignment)"
    echo "   - DO NOT check 'Add a README file'"
    echo "   - DO NOT add .gitignore or license"
    echo ""
    echo "3. Click 'Create repository'"
    echo ""
    echo "4. After creating, run this script again:"
    echo "   ./complete_upload.sh"
    echo ""
    echo "Or manually run:"
    echo "   git push -u origin main"
    echo ""
    echo "=========================================="
    
    # Try to open browser
    if command -v open &> /dev/null; then
        echo ""
        echo "Opening GitHub in your browser..."
        open "https://github.com/new?name=EEP_DL_FinalProject_submission&description=Final+project+for+EEP+596D+-+Predicting+Human+Preference+Between+LLM+Responses+with+Fine-Tuning&public=true"
    fi
    
    exit 0
fi

