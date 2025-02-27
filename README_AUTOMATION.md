# Automated Processing and Backup System

This document explains the automated processing and GitHub backup solution we've implemented.

## Overview

The automation system handles:
1. Processing all existing text files in a GCS bucket
2. Automatically processing new file uploads as they happen
3. Regular backup of code to GitHub
4. Scheduled execution using cron jobs

## Setup Instructions

### 1. Configure GitHub Repository

Run the GitHub setup script:
```bash
./github_setup.sh
```

This script will:
- Initialize a git repository if needed
- Configure your GitHub credentials
- Set up the remote repository
- Make an initial commit and push

### 2. Configure Automated Processing

The `auto_process_bucket.py` script handles:
- Processing all existing files in the bucket
- Setting up a Cloud Run trigger for new uploads
- Pushing code to GitHub

Run it manually:
```bash
python auto_process_bucket.py --bucket=YOUR_BUCKET_NAME --project=YOUR_PROJECT_ID
```

### 3. Configure Automated Backups

The `auto_backup.sh` script handles:
- Checking for changes in your codebase
- Committing and pushing to GitHub

### 4. Set Up Scheduled Jobs

Run the cron setup script:
```bash
./cron_setup.sh
```

This will configure:
- Daily processing of files at 2:00 AM
- Backup to GitHub every 6 hours

## Logging

All activities are logged to:
- `processing_log.txt` - For bucket processing operations
- `backup_log.txt` - For GitHub backup operations

## Manual Operations

### Process Files on Demand

```bash
python auto_process_bucket.py
```

### Backup to GitHub on Demand

```bash
./auto_backup.sh
```

## Troubleshooting

1. If GitHub pushes fail, check your credentials and repository access
2. If file processing fails, check your GCP permissions and bucket access
3. If cron jobs aren't running, check the crontab with `crontab -l`