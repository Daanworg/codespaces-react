#!/bin/bash
# Automated backup script for cloud-rag-webhook repository
# This script can be set up as a cron job to automatically backup code to GitHub

# Change to the project directory
cd "$(dirname "$0")" || exit 1

# Ensure we have the latest code
git pull origin master

# Set timestamp 
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Add all files to git
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes to commit at $TIMESTAMP"
    exit 0
fi

# Commit changes
git commit -m "Automatic backup $TIMESTAMP"

# Push to GitHub
git push origin master

echo "Successfully backed up code to GitHub at $TIMESTAMP"