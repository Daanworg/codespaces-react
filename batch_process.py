#!/usr/bin/env python3
"""
Batch processor for text files in a GCS bucket.
"""

import argparse
import subprocess
import time
from google.cloud import storage

def list_text_files(bucket_name, project_id, limit=None):
    """List text files in a bucket.
    
    Args:
        bucket_name: Name of the GCS bucket.
        project_id: Google Cloud project ID.
        limit: Maximum number of files to return.
        
    Returns:
        List of text file names.
    """
    storage_client = storage.Client(project=project_id)
    blobs = storage_client.list_blobs(bucket_name)
    
    text_files = []
    for blob in blobs:
        if blob.name.lower().endswith('.txt'):
            text_files.append(blob.name)
            if limit and len(text_files) >= limit:
                break
    
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

def main():
    parser = argparse.ArgumentParser(description='Batch process text files in a bucket')
    parser.add_argument('--bucket', default='summary-docs-rag-data-processor', help='GCS bucket name')
    parser.add_argument('--project', default='rag-data-processor', help='Google Cloud project ID')
    parser.add_argument('--dataset', default='summary_dataset', help='BigQuery dataset')
    parser.add_argument('--table', default='rag_chunks', help='BigQuery table')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    # Get list of text files
    print(f"Listing text files in {args.bucket}...")
    files = list_text_files(args.bucket, args.project, args.limit)
    print(f"Found {len(files)} text files")
    
    # Process the files
    process_files(args.bucket, files, args.project, args.dataset, args.table)
    
    print("Batch processing complete!")

if __name__ == "__main__":
    main()