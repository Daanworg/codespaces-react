# Cloud RAG Webhook System

A Retrieval-Augmented Generation (RAG) system built on Google Cloud that processes documents and enables semantic search and question answering.

## Features

- **Automatic Document Processing**: Process text files and PDFs uploaded to a Google Cloud Storage bucket
- **Text Embedding**: Generate vector embeddings for document chunks
- **BigQuery Storage**: Store document text, summaries, and vector embeddings
- **RAG Querying**: Perform semantic search and retrieval-augmented generation
- **Web Interface**: Access document processing and querying via web UI
- **GitHub Backup**: Automatically backup code to GitHub
- **Cloud Triggers**: Process new files automatically when uploaded

## Architecture

This system uses the following Google Cloud components:

- **Cloud Storage**: For storing input documents
- **Vertex AI**: For generating embeddings and text generation
- **DocumentAI**: For processing PDF documents
- **BigQuery**: For storing processed data with vector search capabilities
- **Cloud Run**: For hosting web services
- **Eventarc**: For triggering processing on new uploads

## Setup Instructions

### Prerequisites

- Google Cloud account with access to:
  - Cloud Storage
  - Vertex AI
  - DocumentAI
  - BigQuery
  - Cloud Run
  - Eventarc
- Python 3.10+
- Git repository access

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Daanworg/cloud-rag-webhook.git
   cd cloud-rag-webhook
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   export GOOGLE_CLOUD_PROJECT=rag-data-processor
   export DOCAI_PROCESSOR=your-processor-id
   export DOCAI_LOCATION=us
   export OUTPUT_BUCKET=summary-docs-rag-data-processor
   export BQ_DATASET=summary_dataset
   export BQ_TABLE=summaries
   export BQ_RAG_TABLE=rag_chunks
   export GITHUB_BACKUP_ENABLED=true
   ```

### Deployment

1. Deploy to Cloud Run:
   ```
   ./deploy_rag.sh --project=YOUR_PROJECT_ID --region=YOUR_REGION
   ```

2. Set up an event trigger for file uploads:
   ```
   ./create_trigger.sh
   ```

3. Process all existing documents in a bucket:
   ```
   ./auto_process_bucket.py --bucket=YOUR_BUCKET_NAME --project=YOUR_PROJECT_ID
   ```

## Automatic Processing

### Processing All Files in a Bucket

To process all existing files in a bucket and set up triggers for future uploads:

```
./auto_process_bucket.py --bucket=summary-docs-rag-data-processor --project=rag-data-processor
```

This script:
1. Processes all text files in the bucket
2. Sets up a trigger for future uploads
3. Backs up code to GitHub

### GitHub Backup

To manually backup all code to GitHub:

```
./auto_backup.sh
```

To enable automatic GitHub backup when files are processed, set:

```
export GITHUB_BACKUP_ENABLED=true
```

## Querying the System

You can query the system through the web interface at:
https://cloud-rag-webhook-164738887219.us-central1.run.app/

## Development

### Running Locally

```
python app.py
```

### Updating the Cloud Function

To update the Cloud Function with the latest code:

```
# Replace main.py with the updated version
cp main_updated.py main.py

# Deploy the updated Cloud Function
./deploy_rag.sh --project=YOUR_PROJECT_ID --region=YOUR_REGION
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
