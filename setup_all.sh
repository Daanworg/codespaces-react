#!/bin/bash
# Master script to set up the entire automated process

# Display banner
echo "=================================================="
echo "   Cloud RAG Webhook - Automation Setup Script    "
echo "=================================================="
echo ""

# Step 1: Check dependencies
echo "Checking dependencies..."
MISSING_DEPS=0

# Check for Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is not installed."
    MISSING_DEPS=1
else
    echo "✅ Python 3 is installed."
fi

# Check for Git
if ! command -v git &>/dev/null; then
    echo "❌ Git is not installed."
    MISSING_DEPS=1
else
    echo "✅ Git is installed."
fi

# Check for Google Cloud SDK
if ! command -v gcloud &>/dev/null; then
    echo "❌ Google Cloud SDK is not installed."
    MISSING_DEPS=1
else
    echo "✅ Google Cloud SDK is installed."
fi

# Exit if dependencies are missing
if [ $MISSING_DEPS -eq 1 ]; then
    echo "Please install missing dependencies and try again."
    exit 1
fi

# Step 2: Install Python requirements
echo -e "\nInstalling Python requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python requirements."
    exit 1
else
    echo "✅ Python requirements installed successfully."
fi

# Step 3: GitHub setup
echo -e "\nSetting up GitHub repository..."
echo "This step configures your git repository and connects it to GitHub."
read -p "Do you want to proceed with GitHub setup? (y/n): " GIT_SETUP
if [[ $GIT_SETUP == "y" ]]; then
    ./github_setup.sh
    if [ $? -ne 0 ]; then
        echo "❌ GitHub setup encountered an issue."
        echo "You can run './github_setup.sh' separately later."
    else
        echo "✅ GitHub repository configured successfully."
    fi
else
    echo "Skipping GitHub setup."
fi

# Step 4: Set up cron jobs
echo -e "\nSetting up scheduled tasks (cron jobs)..."
echo "This will schedule automatic processing and backups."
read -p "Do you want to set up scheduled tasks? (y/n): " CRON_SETUP
if [[ $CRON_SETUP == "y" ]]; then
    ./cron_setup.sh
    if [ $? -ne 0 ]; then
        echo "❌ Cron job setup encountered an issue."
        echo "You can run './cron_setup.sh' separately later."
    else
        echo "✅ Scheduled tasks configured successfully."
    fi
else
    echo "Skipping cron job setup."
fi

# Step 5: Initial processing
echo -e "\nInitial bucket processing..."
echo "This will process all existing files in your bucket and set up triggers for new uploads."
read -p "Do you want to run the initial processing now? (y/n): " INITIAL_PROCESS
if [[ $INITIAL_PROCESS == "y" ]]; then
    echo "Enter your bucket details:"
    read -p "GCS bucket name [summary-docs-rag-data-processor]: " BUCKET_NAME
    BUCKET_NAME=${BUCKET_NAME:-"summary-docs-rag-data-processor"}
    
    read -p "GCP project ID [rag-data-processor]: " PROJECT_ID
    PROJECT_ID=${PROJECT_ID:-"rag-data-processor"}
    
    python auto_process_bucket.py --bucket="$BUCKET_NAME" --project="$PROJECT_ID"
    if [ $? -ne 0 ]; then
        echo "❌ Initial processing encountered an issue."
    else
        echo "✅ Initial processing completed successfully."
    fi
else
    echo "Skipping initial processing."
fi

# All done
echo -e "\n=================================================="
echo "            Setup Complete!                       "
echo "=================================================="
echo ""
echo "The automation system is now configured."
echo "  - Check README_AUTOMATION.md for usage details"
echo "  - View processing_log.txt for processing logs"
echo "  - View backup_log.txt for backup logs"
echo ""
echo "To run processes manually:"
echo "  - Process bucket: python auto_process_bucket.py"
echo "  - Backup code: ./auto_backup.sh"
echo ""
echo "Thanks for using the Cloud RAG Webhook automation setup!"