#!/usr/bin/env python3
"""
Automatic processor for all text files in a GCS bucket.
- Processes all existing files
- Configures a trigger for new uploads
- Pushes code to GitHub
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from google.cloud import storage

def list_all_text_files(bucket_name, project_id):
    """List all text files in a bucket.
    
    Args:
        bucket_name: Name of the GCS bucket.
        project_id: Google Cloud project ID.
        
    Returns:
        List of text file names.
    """
    storage_client = storage.Client(project=project_id)
    blobs = storage_client.list_blobs(bucket_name)
    
    text_files = []
    for blob in blobs:
        if blob.name.lower().endswith('.txt'):
            text_files.append(blob.name)
    
    return text_files

def process_files(bucket_name, file_list, project_id, dataset, table):
    """Process a list of files.
    
    Args:
        bucket_name: Name of the GCS bucket.
        file_list: List of file names to process.
        project_id: Google Cloud project ID.
        dataset: BigQuery dataset name.
        table: BigQuery table name.
    """
    total = len(file_list)
    
    for i, filename in enumerate(file_list):
        print(f"Processing file {i+1}/{total}: {filename}")
        
        # Call the process_text.py script
        cmd = [
            "python", "process_text.py",
            "--bucket", bucket_name,
            "--file=" + filename,  # Use equals sign to handle filenames with dashes
            "--project", project_id,
            "--dataset", dataset,
            "--table", table
        ]
        
        # Use subprocess to run the command
        try:
            env = {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/docai-key.json"}
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(f"Success: {filename}")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
            print(e.stdout)
            print(e.stderr)
        
        # Pause between files to avoid rate limiting
        time.sleep(2)

def setup_trigger(project_id, bucket_name, service_name, region="us-central1"):
    """Set up an Eventarc trigger for new file uploads.
    
    Args:
        project_id: Google Cloud project ID.
        bucket_name: Name of the GCS bucket.
        service_name: Cloud Run service name.
        region: Cloud region.
    """
    trigger_name = f"{service_name}-trigger"
    
    # Check if the trigger already exists
    check_cmd = f"gcloud eventarc triggers describe {trigger_name} --location={region} --project={project_id} 2>/dev/null || echo 'Trigger not found'"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    
    if "Trigger not found" in result.stdout:
        print(f"Creating new trigger: {trigger_name}")
        
        # Create the trigger
        cmd = [
            "gcloud", "eventarc", "triggers", "create", trigger_name,
            f"--location={region}",
            f"--service-account=summary-trigger-sa@{project_id}.iam.gserviceaccount.com",
            f"--destination-run-service={service_name}",
            f"--destination-run-region={region}",
            f"--event-filters=bucket={bucket_name}",
            "--event-filters=type=google.cloud.storage.object.v1.finalized",
            f"--project={project_id}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created trigger: {trigger_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating trigger: {e}")
            return False
    else:
        print(f"Trigger already exists: {trigger_name}")
    
    return True

def push_to_github():
    """Push all code to GitHub for backup."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Make sure we're in a git repository
        subprocess.run(["git", "status"], check=True, stdout=subprocess.PIPE)
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit changes
        commit_msg = f"Automatic backup {timestamp}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # Push to GitHub
        subprocess.run(["git", "push", "origin", "master"], check=True)
        
        print(f"Successfully pushed to GitHub at {timestamp}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to GitHub: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process all text files and set up automation')
    parser.add_argument('--bucket', default='summary-docs-rag-data-processor', help='GCS bucket name')
    parser.add_argument('--project', default='rag-data-processor', help='Google Cloud project ID')
    parser.add_argument('--dataset', default='summary_dataset', help='BigQuery dataset')
    parser.add_argument('--table', default='rag_chunks', help='BigQuery table')
    parser.add_argument('--service', default='cloud-rag-webhook', help='Cloud Run service name')
    parser.add_argument('--region', default='us-central1', help='Cloud region')
    parser.add_argument('--skip-processing', action='store_true', help='Skip processing existing files')
    parser.add_argument('--skip-trigger', action='store_true', help='Skip setting up the trigger')
    parser.add_argument('--skip-github', action='store_true', help='Skip GitHub backup')
    
    args = parser.parse_args()
    
    # 1. Process all existing files in the bucket
    if not args.skip_processing:
        print(f"Listing all text files in {args.bucket}...")
        files = list_all_text_files(args.bucket, args.project)
        print(f"Found {len(files)} text files")
        
        if files:
            process_files(args.bucket, files, args.project, args.dataset, args.table)
        else:
            print("No text files found in the bucket.")
    
    # 2. Set up trigger for new file uploads
    if not args.skip_trigger:
        print(f"Setting up trigger for bucket {args.bucket}...")
        setup_trigger(args.project, args.bucket, args.service, args.region)
    
    # 3. Push all code to GitHub
    if not args.skip_github:
        print("Pushing code to GitHub...")
        push_to_github()
    
    print("Automation setup complete!")

if __name__ == "__main__":
    main()