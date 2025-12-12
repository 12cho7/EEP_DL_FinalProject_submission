#!/bin/bash

# Verify and Push Script
# Run this after creating the GitHub repository

echo "=========================================="
echo "Verifying and Pushing to GitHub"
echo "=========================================="
echo ""

# Check current status
echo "Current Git status:"
git status --short
echo ""

# Check remote
echo "Remote configuration:"
git remote -v
echo ""

# Try to push
echo "Attempting to push to GitHub..."
echo ""

if git push -u origin main 2>&1; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Upload completed!"
    echo "=========================================="
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/12cho7/EEP_DL_FinalProject_submission"
    echo ""
    echo "Verifying upload..."
    sleep 3
    
    # Try to verify by checking if we can fetch
    if git ls-remote --exit-code origin main &>/dev/null; then
        echo "✓ Repository verified!"
        echo ""
        echo "Files uploaded:"
        git ls-tree -r main --name-only | head -20
        echo ""
        echo "Total files: $(git ls-tree -r main --name-only | wc -l | tr -d ' ')"
        echo ""
        echo "=========================================="
        echo "✅ All done! Ready for submission!"
        echo "=========================================="
        echo ""
        echo "Submit this URL to your professor:"
        echo "https://github.com/12cho7/EEP_DL_FinalProject_submission"
        echo ""
        
        # Open browser to verify
        if command -v open &> /dev/null; then
            echo "Opening repository in browser..."
            open "https://github.com/12cho7/EEP_DL_FinalProject_submission"
        fi
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Push failed!"
    echo "=========================================="
    echo ""
    echo "Possible reasons:"
    echo "1. Repository not created yet - please create it first at https://github.com/new"
    echo "2. Authentication required - you may need to:"
    echo "   - Use Personal Access Token (PAT)"
    echo "   - Or configure GitHub CLI: gh auth login"
    echo ""
    echo "After fixing, run this script again:"
    echo "   ./verify_and_push.sh"
    echo ""
    exit 1
fi

