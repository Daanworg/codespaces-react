#!/bin/bash

# Create an Eventarc trigger for the cloud-rag-webhook service
gcloud eventarc triggers create cloud-rag-webhook-trigger \
  --location=us-central1 \
  --service-account=summary-trigger-sa@rag-data-processor.iam.gserviceaccount.com \
  --destination-run-service=cloud-rag-webhook \
  --destination-run-region=us-central1 \
  --event-filters="bucket=summary-docs-rag-data-processor" \
  --event-filters="type=google.cloud.storage.object.v1.finalized"

# Add labels (optional)
gcloud eventarc triggers update cloud-rag-webhook-trigger \
  --location=us-central1 \
  --update-labels="goog-ccm=true,goog-solutions-console-deployment-name=generative-ai-document-summarization,goog-solutions-console-solution-id=generative-ai-document-summarization"

echo "Trigger creation completed."