#!/bin/bash
# Deploy script for the RAG system

# Exit on error
set -e

# Default variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_ACCOUNT=""
DATASET_ID="document_processing"
RAG_TABLE="rag_chunks"
MAIN_TABLE="document_summaries"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --service-account)
      SERVICE_ACCOUNT="$2"
      shift 2
      ;;
    --dataset)
      DATASET_ID="$2"
      shift 2
      ;;
    --rag-table)
      RAG_TABLE="$2"
      shift 2
      ;;
    --main-table)
      MAIN_TABLE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "🚀 Deploying RAG system..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Dataset: $DATASET_ID"
echo "RAG Table: $RAG_TABLE"
echo "Main Table: $MAIN_TABLE"

# Check if service account is provided
if [ -z "$SERVICE_ACCOUNT" ]; then
  echo "⚠️ No service account provided, using default compute service account"
  SERVICE_ACCOUNT="$(gcloud iam service-accounts list --filter="name:compute" --format="value(email)" --limit=1)"
fi
echo "Service Account: $SERVICE_ACCOUNT"

# Make sure BigQuery dataset exists
echo "🗂️ Checking BigQuery dataset..."
if ! bq ls --project_id=$PROJECT_ID "$DATASET_ID" &>/dev/null; then
  echo "Creating dataset $DATASET_ID"
  bq --location=$REGION mk \
    --dataset \
    --description="Dataset for document processing and RAG system" \
    "${PROJECT_ID}:${DATASET_ID}"
else
  echo "Dataset $DATASET_ID already exists"
fi

# Create or update the RAG table schema
echo "📊 Setting up RAG table schema..."
# Using a temporary JSON schema file
cat > /tmp/rag_schema.json << EOF
[
  {
    "name": "chunk_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Unique identifier for the chunk"
  },
  {
    "name": "document_path",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Path to the source document in Cloud Storage"
  },
  {
    "name": "event_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "ID of the event that triggered processing"
  },
  {
    "name": "time_processed",
    "type": "TIMESTAMP",
    "mode": "REQUIRED",
    "description": "When the chunk was processed"
  },
  {
    "name": "text_chunk",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "The actual text content of the chunk"
  },
  {
    "name": "vector_embedding",
    "type": "FLOAT64",
    "mode": "REPEATED",
    "description": "Vector embedding of the text chunk"
  },
  {
    "name": "metadata",
    "type": "JSON",
    "mode": "NULLABLE",
    "description": "Additional metadata about the chunk"
  },
  {
    "name": "questions",
    "type": "STRING",
    "mode": "REPEATED",
    "description": "Sample questions this chunk can answer"
  },
  {
    "name": "answers",
    "type": "STRING",
    "mode": "REPEATED",
    "description": "Answers to the sample questions"
  },
  {
    "name": "category",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Category or topic of the chunk"
  },
  {
    "name": "keywords",
    "type": "STRING",
    "mode": "REPEATED",
    "description": "Important keywords from the chunk"
  }
]
EOF

# Check if table exists, create it if it doesn't
if ! bq ls --project_id=$PROJECT_ID "${DATASET_ID}.${RAG_TABLE}" &>/dev/null; then
  echo "Creating table ${RAG_TABLE}"
  bq mk \
    --table \
    --clustering_fields="category" \
    --description="RAG chunks for document search and retrieval" \
    "${PROJECT_ID}:${DATASET_ID}.${RAG_TABLE}" \
    /tmp/rag_schema.json
else
  echo "Table ${RAG_TABLE} already exists, updating schema"
  bq update \
    --clustering_fields="category" \
    "${PROJECT_ID}:${DATASET_ID}.${RAG_TABLE}" \
    /tmp/rag_schema.json
fi

# Clean up temp file
rm /tmp/rag_schema.json

# Deploy the Cloud Function
echo "☁️ Deploying Cloud Function for document processing..."
gcloud functions deploy document-processor \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=on_cloud_event \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=YOUR_INPUT_BUCKET" \
  --service-account=$SERVICE_ACCOUNT \
  --set-env-vars="BQ_DATASET=${DATASET_ID},BQ_TABLE=${MAIN_TABLE},BQ_RAG_TABLE=${RAG_TABLE},DOCAI_PROCESSOR=YOUR_DOCAI_PROCESSOR_ID,OUTPUT_BUCKET=YOUR_OUTPUT_BUCKET"

echo "✅ Deployment complete!"
echo ""
echo "To use the RAG query system:"
echo "1. Set environment variables:"
echo "   export BQ_DATASET=${DATASET_ID}"
echo "   export BQ_RAG_TABLE=${RAG_TABLE}"
echo ""
echo "2. Run the query interface:"
echo "   python rag_server.py"
echo ""
echo "3. Or query directly from the command line:"
echo "   python rag_query.py 'How do I create a Cloud Storage bucket?'"