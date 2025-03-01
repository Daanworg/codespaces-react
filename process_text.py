#!/usr/bin/env python3
"""
Direct text file processor for RAG system.
Processes text files in a bucket without using DocumentAI.
"""

import os
import re
import uuid
import json
import datetime
from typing import List, Dict, Any
import argparse

from google.cloud import storage
from google.cloud import bigquery
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel

# Constants
CHUNK_SIZE = 2000  # Maximum characters per chunk
CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks

# For generating sample questions and answers for each chunk
QA_GENERATION_PROMPT = """
Based on the following text, generate 3 relevant questions that this text can answer, 
and provide accurate answers to those questions based only on the information contained in this text.
Format your response as a JSON array of objects, each with 'question' and 'answer' keys.

TEXT:
{text}
"""

def chunk_text(text: str) -> List[str]:
    """Split text into chunks of appropriate size for RAG.
    
    Args:
        text: The text to split into chunks.
        
    Returns:
        A list of text chunks.
    """
    # Split into paragraphs first
    paragraphs = [p for p in text.split("\n") if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph itself is too large, split it into sentences
        if len(paragraph) > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
                    current_chunk += (sentence + " ")
                else:
                    # Save current chunk if it's not empty
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk
                    # If this sentence is too large, we'll need to split it
                    if len(sentence) > CHUNK_SIZE:
                        # Split long sentence into parts
                        for i in range(0, len(sentence), CHUNK_SIZE - CHUNK_OVERLAP):
                            chunks.append(sentence[i:i + CHUNK_SIZE].strip())
                    else:
                        current_chunk = sentence + " "
        else:
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"
            else:
                current_chunk += (paragraph + "\n")
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Ensure chunks have some overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1:
            # Add some overlap from next chunk if there is one
            next_chunk = chunks[i + 1]
            if len(chunk) < CHUNK_SIZE - CHUNK_OVERLAP:
                overlap = min(CHUNK_OVERLAP, len(next_chunk))
                chunk += " " + next_chunk[:overlap]
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks

def generate_qa_pairs(text: str, model_name: str = "gemini-pro") -> List[Dict[str, str]]:
    """Generate question-answer pairs for a chunk of text.
    
    Args:
        text: The text to generate QA pairs for.
        model_name: The name of the model to use.
        
    Returns:
        A list of dictionaries with 'question' and 'answer' keys.
    """
    try:
        model = GenerativeModel(model_name)
        prompt = QA_GENERATION_PROMPT.format(text=text)
        result = model.generate_content(prompt)
        
        # Process the result as JSON
        try:
            qa_pairs = json.loads(result.text)
            return qa_pairs
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            print(f"Warning: QA response was not valid JSON: {result.text[:100]}...")
            return [{"question": "What is this document about?", 
                     "answer": "The document contains technical information."}]
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return [{"question": "What is this document about?", 
                 "answer": "The document contains technical information."}]

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from the text.
    
    Args:
        text: The text to extract keywords from.
        
    Returns:
        A list of keywords.
    """
    # Simple extraction based on frequency and filtering
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'from'])
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 keywords
    return [word for word, count in sorted_words[:10]]

def determine_category(text: str, model_name: str = "gemini-pro") -> str:
    """Determine the category of the text.
    
    Args:
        text: The text to categorize.
        model_name: The model to use for categorization.
        
    Returns:
        The category as a string.
    """
    try:
        model = GenerativeModel(model_name)
        prompt = "Analyze this text and provide a single category label that best describes it (like 'Cloud Storage', 'Machine Learning', 'Security', etc.):\n\n" + text[:1000]
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        print(f"Error determining category: {e}")
        return "Uncategorized"

def write_rag_to_bigquery(
    rows: List[Dict[str, Any]],
    bq_dataset: str,
    bq_table: str,
    project_id: str
) -> None:
    """Write RAG data to BigQuery.
    
    Args:
        rows: List of row dictionaries to insert.
        bq_dataset: Name of the BigQuery dataset.
        bq_table: Name of the BigQuery table.
        project_id: Google Cloud project ID.
    """
    bq_client = bigquery.Client(project=project_id)
    
    # Make sure the table exists
    try:
        table_ref = f"{project_id}.{bq_dataset}.{bq_table}"
        bq_client.get_table(table_ref)
        print(f"Table {table_ref} exists")
    except Exception:
        # Table doesn't exist, create it
        schema = [
            bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("document_path", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("time_processed", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("text_chunk", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("vector_embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("questions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("answers", "STRING", mode="REPEATED"),
            bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("keywords", "STRING", mode="REPEATED"),
        ]
        
        table = bigquery.Table(table_ref, schema=schema)
        # Enable search capabilities
        table.clustering_fields = ["category"]
        bq_client.create_table(table)
        print(f"Created table {table_ref}")
    
    # Convert datetime objects to strings for JSON serialization
    for row in rows:
        if isinstance(row.get('time_processed'), datetime.datetime):
            row['time_processed'] = row['time_processed'].isoformat()
        if isinstance(row.get('metadata'), dict):
            row['metadata'] = json.dumps(row['metadata'])
    
    # Insert rows
    errors = bq_client.insert_rows_json(f"{project_id}.{bq_dataset}.{bq_table}", rows)
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print(f"Successfully inserted {len(rows)} rows into {bq_table}")

def process_text_file(
    bucket_name: str,
    filename: str,
    event_id: str,
    bq_dataset: str,
    bq_rag_table: str,
    project_id: str
) -> None:
    """Process a text file from GCS and save results to BigQuery.
    
    Args:
        bucket_name: The name of the GCS bucket.
        filename: The name of the file to process.
        event_id: ID of the triggering event.
        bq_dataset: BigQuery dataset name.
        bq_rag_table: BigQuery RAG table name.
        project_id: Google Cloud project ID.
    """
    # Initialize Vertex AI with the project ID
    vertexai.init(project=project_id, location="us-central1")
    
    print(f"Processing {filename} from {bucket_name}")
    
    # Download the file
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    
    try:
        # Download and decode the text
        text_content = blob.download_as_text()
        print(f"Downloaded text file: {len(text_content)} characters")
        
        # Chunk the text
        chunks = chunk_text(text_content)
        print(f"Created {len(chunks)} chunks")
        
        # Get embedding model
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        # Process each chunk
        rag_rows = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Generate embeddings
            embedding = embedding_model.get_embeddings([chunk])[0].values
            
            # Generate questions and answers
            qa_pairs = generate_qa_pairs(chunk)
            
            # Extract keywords
            keywords = extract_keywords(chunk)
            
            # Determine category
            category = determine_category(chunk)
            
            # Create a unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Add to rows
            rag_rows.append({
                "chunk_id": chunk_id,
                "document_path": f"gs://{bucket_name}/{filename}",
                "event_id": event_id,
                "time_processed": datetime.datetime.now(),
                "text_chunk": chunk,
                "vector_embedding": embedding,
                "metadata": {"source": filename, "chunk_number": i, "chunk_total": len(chunks)},
                "questions": [qa["question"] for qa in qa_pairs],
                "answers": [qa["answer"] for qa in qa_pairs],
                "category": category,
                "keywords": keywords
            })
        
        # Write to BigQuery
        write_rag_to_bigquery(rag_rows, bq_dataset, bq_rag_table, project_id)
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process a text file for RAG')
    parser.add_argument('--bucket', required=True, help='GCS bucket name')
    parser.add_argument('--file', required=True, help='Filename to process')
    parser.add_argument('--event-id', default=f'manual-{uuid.uuid4()}', help='Event ID')
    parser.add_argument('--dataset', default='summary_dataset', help='BigQuery dataset')
    parser.add_argument('--table', default='rag_chunks', help='BigQuery table')
    parser.add_argument('--project', required=True, help='Google Cloud project ID')
    
    args = parser.parse_args()
    
    process_text_file(
        bucket_name=args.bucket,
        filename=args.file,
        event_id=args.event_id,
        bq_dataset=args.dataset,
        bq_rag_table=args.table,
        project_id=args.project
    )

if __name__ == "__main__":
    main()