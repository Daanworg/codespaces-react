# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import uuid
from collections.abc import Iterator, List
from datetime import datetime
from typing import Dict, Any

import functions_framework
from cloudevents.http import CloudEvent
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import bigquery
from google.cloud import storage  # type: ignore
from vertexai.preview.generative_models import GenerativeModel  # type: ignore
from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore

SUMMARIZATION_PROMPT = """\
Give me a summary of the following text.
Use simple language and give examples.
Explain to an undergraduate.

TEXT:
{text}
"""

CHUNK_SIZE = 2000  # Maximum characters per chunk
CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks

# For generating sample questions and answers for each chunk
QA_GENERATION_PROMPT = """\
Based on the following text, generate 3 relevant questions that this text can answer, 
and provide accurate answers to those questions based only on the information contained in this text.
Format your response as a JSON array of objects, each with 'question' and 'answer' keys.

TEXT:
{text}
"""


@functions_framework.cloud_event
def on_cloud_event(event: CloudEvent) -> None:
    """Process a new document from an Eventarc event.

    Args:
        event: CloudEvent object.
    """
    try:
        process_document(
            event_id=event.data["id"],
            input_bucket=event.data["bucket"],
            filename=event.data["name"],
            mime_type=event.data["contentType"],
            time_uploaded=datetime.fromisoformat(event.data["timeCreated"]),
            docai_processor_id=os.environ["DOCAI_PROCESSOR"],
            docai_location=os.environ.get("DOCAI_LOCATION", "us"),
            output_bucket=os.environ["OUTPUT_BUCKET"],
            bq_dataset=os.environ["BQ_DATASET"],
            bq_table=os.environ["BQ_TABLE"],
            bq_rag_table=os.environ.get("BQ_RAG_TABLE", "rag_chunks"),
        )
    except Exception as e:
        logging.exception(e, stack_info=True)


def process_document(
    event_id: str,
    input_bucket: str,
    filename: str,
    mime_type: str,
    time_uploaded: datetime,
    docai_processor_id: str,
    docai_location: str,
    output_bucket: str,
    bq_dataset: str,
    bq_table: str,
    bq_rag_table: str,
):
    """Process a new document.

    Args:
        event_id: ID of the event.
        input_bucket: Name of the input bucket.
        filename: Name of the input file.
        mime_type: MIME type of the input file.
        time_uploaded: Time the input file was uploaded.
        docai_processor_id: ID of the Document AI processor.
        docai_location: Location of the Document AI processor.
        output_bucket: Name of the output bucket.
        bq_dataset: Name of the BigQuery dataset.
        bq_table: Name of the BigQuery table.
        bq_rag_table: Name of the BigQuery RAG chunks table.
    """
    doc_path = f"gs://{input_bucket}/{filename}"
    print(f"📖 {event_id}: Getting document text")
    doc_text = "\n".join(
        get_document_text(
            doc_path,
            mime_type,
            docai_processor_id,
            output_bucket,
            docai_location,
        )
    )

    model_name = "gemini-pro"
    print(f"📝 {event_id}: Summarizing document with {model_name}")
    print(f"  - Text length:    {len(doc_text)} characters")
    doc_summary = generate_summary(doc_text, model_name)
    print(f"  - Summary length: {len(doc_summary)} characters")

    print(f"🗃️ {event_id}: Writing document summary to BigQuery: {bq_dataset}.{bq_table}")
    write_to_bigquery(
        event_id=event_id,
        time_uploaded=time_uploaded,
        doc_path=doc_path,
        doc_text=doc_text,
        doc_summary=doc_summary,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
    )
    
    # Process for RAG
    print(f"🧩 {event_id}: Chunking text for RAG")
    chunks = chunk_text(doc_text)
    print(f"  - Created {len(chunks)} chunks")
    
    # Get embedding model
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    # Process each chunk
    rag_rows = []
    for i, chunk in enumerate(chunks):
        print(f"  - Processing chunk {i+1}/{len(chunks)}")
        
        # Generate embeddings
        embedding = embedding_model.get_embeddings([chunk])[0].values
        
        # Generate questions and answers
        qa_pairs = generate_qa_pairs(chunk, model_name)
        
        # Extract keywords (simple method, could be improved)
        keywords = extract_keywords(chunk)
        
        # Determine category (simple method, could be improved)
        category = determine_category(chunk, model_name)
        
        # Create a unique ID for this chunk
        chunk_id = str(uuid.uuid4())
        
        # Add to rows
        rag_rows.append({
            "chunk_id": chunk_id,
            "document_path": doc_path,
            "event_id": event_id,
            "time_processed": datetime.now(),
            "text_chunk": chunk,
            "vector_embedding": embedding,
            "metadata": {
                "source": filename,
                "chunk_number": i,
                "chunk_total": len(chunks)
            },
            "questions": [qa["question"] for qa in qa_pairs],
            "answers": [qa["answer"] for qa in qa_pairs],
            "category": category,
            "keywords": keywords
        })
    
    print(f"🗃️ {event_id}: Writing RAG chunks to BigQuery: {bq_dataset}.{bq_rag_table}")
    write_rag_to_bigquery(
        rows=rag_rows,
        bq_dataset=bq_dataset,
        bq_table=bq_rag_table,
    )

    print(f"✅ {event_id}: Done!")


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
        import json
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
    # This could be improved with more sophisticated NLP techniques
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words (this is a simple list, could be expanded)
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


def get_document_text(
    input_file: str,
    mime_type: str,
    processor_id: str,
    temp_bucket: str,
    docai_location: str = "us",
) -> Iterator[str]:
    """Perform Optical Character Recognition (OCR) with Document AI on a Cloud Storage file.

    For more information, see:
        https://cloud.google.com/document-ai/docs/process-documents-ocr

    Args:
        input_file: GCS URI of the document file.
        mime_type: MIME type of the document file.
        processor_id: ID of the Document AI processor.
        temp_bucket: GCS bucket to store Document AI temporary files.
        docai_location: Location of the Document AI processor.

    Yields: The document text chunks.
    """
    # You must set the `api_endpoint` if you use a location other than "us".
    documentai_client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{docai_location}-documentai.googleapis.com")
    )

    # We're using batch_process_documents instead of process_document because
    # process_document has a quota limit of 15 pages per document, while
    # batch_process_documents has a quota limit of 500 pages per request.
    #   https://cloud.google.com/document-ai/quotas#general_processors
    operation = documentai_client.batch_process_documents(
        request=documentai.BatchProcessRequest(
            name=processor_id,
            input_documents=documentai.BatchDocumentsInputConfig(
                gcs_documents=documentai.GcsDocuments(
                    documents=[
                        documentai.GcsDocument(gcs_uri=input_file, mime_type=mime_type),
                    ],
                ),
            ),
            document_output_config=documentai.DocumentOutputConfig(
                gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                    gcs_uri=f"gs://{temp_bucket}/ocr/{input_file.split('gs://')[-1]}",
                ),
            ),
        ),
    )
    operation.result()

    # Read the results of the Document AI operation from Cloud Storage.
    storage_client = storage.Client()
    metadata = documentai.BatchProcessMetadata(operation.metadata)
    output_gcs_path = metadata.individual_process_statuses[0].output_gcs_destination
    (output_bucket, output_prefix) = output_gcs_path.removeprefix("gs://").split("/", 1)
    for blob in storage_client.list_blobs(output_bucket, prefix=output_prefix):
        blob_contents = blob.download_as_bytes()
        document = documentai.Document.from_json(blob_contents, ignore_unknown_fields=True)
        yield document.text


def generate_summary(text: str, model_name: str = "gemini-pro") -> str:
    """Generate a summary of the given text.

    Args:
        text: The text to summarize.
        model_name: The name of the model to use for summarization.

    Returns:
        The generated summary.
    """
    model = GenerativeModel(model_name)
    prompt = SUMMARIZATION_PROMPT.format(text=text)
    return model.generate_content(prompt).text


def write_to_bigquery(
    event_id: str,
    time_uploaded: datetime,
    doc_path: str,
    doc_text: str,
    doc_summary: str,
    bq_dataset: str,
    bq_table: str,
) -> None:
    """Write the summary to BigQuery.

    Args:
        event_id: The Eventarc trigger event ID.
        time_uploaded: Time the document was uploaded.
        doc_path: Cloud Storage path to the document.
        doc_text: Text extracted from the document.
        doc_summary: Summary generated fro the document.
        bq_dataset: Name of the BigQuery dataset.
        bq_table: Name of the BigQuery table.
    """
    bq_client = bigquery.Client()
    bq_client.insert_rows(
        table=bq_client.get_table(f"{bq_dataset}.{bq_table}"),
        rows=[
            {
                "event_id": event_id,
                "time_uploaded": time_uploaded,
                "time_processed": datetime.now(),
                "document_path": doc_path,
                "document_text": doc_text,
                "document_summary": doc_summary,
            },
        ],
    )


def write_rag_to_bigquery(
    rows: List[Dict[str, Any]],
    bq_dataset: str,
    bq_table: str,
) -> None:
    """Write RAG data to BigQuery.
    
    Args:
        rows: List of row dictionaries to insert.
        bq_dataset: Name of the BigQuery dataset.
        bq_table: Name of the BigQuery table.
    """
    bq_client = bigquery.Client()
    
    # Make sure the table exists
    try:
        table_ref = f"{bq_dataset}.{bq_table}"
        bq_client.get_table(table_ref)
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
    
    # Insert rows
    errors = bq_client.insert_rows_json(f"{bq_dataset}.{bq_table}", rows)
    if errors:
        print(f"Errors inserting rows: {errors}")