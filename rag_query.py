# RAG Query System for Google Cloud Documentation
import os
from typing import List, Dict, Any

from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel

class RAGQuerySystem:
    """A system for retrieving and answering questions using the RAG approach."""
    
    def __init__(
        self, 
        bq_dataset: str,
        bq_rag_table: str,
        embedding_model_name: str = "text-embedding-004",
        generation_model_name: str = "gemini-pro",
        top_k: int = 5
    ):
        """Initialize the RAG query system.
        
        Args:
            bq_dataset: The BigQuery dataset containing the RAG chunks.
            bq_rag_table: The BigQuery table containing the RAG chunks.
            embedding_model_name: The name of the embedding model to use.
            generation_model_name: The name of the LLM to use for generation.
            top_k: The number of chunks to retrieve.
        """
        self.bq_client = bigquery.Client()
        self.bq_dataset = bq_dataset
        self.bq_rag_table = bq_rag_table
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        self.generation_model = GenerativeModel(generation_model_name)
        self.top_k = top_k
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The question to answer.
            
        Returns:
            A dictionary containing the answer and relevant chunks.
        """
        # Generate embedding for the question
        query_embedding = self.embedding_model.get_embeddings([question])[0].values
        
        # Retrieve relevant chunks from BigQuery using vector similarity
        relevant_chunks = self._retrieve_relevant_chunks(query_embedding)
        
        # Generate answer using retrieved chunks
        answer = self._generate_answer(question, relevant_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": relevant_chunks
        }
    
    def _retrieve_relevant_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks from BigQuery.
        
        Args:
            query_embedding: The embedding of the query.
            
        Returns:
            A list of relevant chunk dictionaries.
        """
        table_name = f"{self.bq_dataset}.{self.bq_rag_table}"
        
        # This SQL calculates the cosine similarity between the query embedding
        # and each document embedding in the database
        query = f"""
        SELECT
            chunk_id,
            document_path,
            text_chunk,
            category,
            keywords,
            questions,
            answers,
            metadata,
            -- Calculate cosine similarity
            (
                SELECT SUM(a * b) / SQRT(SUM(a * a) * SUM(b * b))
                FROM UNNEST(vector_embedding) a WITH OFFSET pos
                JOIN UNNEST(@query_embedding) b WITH OFFSET pos
                USING (pos)
            ) AS similarity
        FROM
            `{table_name}`
        ORDER BY
            similarity DESC
        LIMIT
            {self.top_k}
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding)
            ]
        )
        
        query_job = self.bq_client.query(query, job_config=job_config)
        results = query_job.result()
        
        chunks = []
        for row in results:
            chunks.append({
                "chunk_id": row.chunk_id,
                "document_path": row.document_path,
                "text_chunk": row.text_chunk,
                "category": row.category,
                "keywords": row.keywords,
                "questions": row.questions,
                "answers": row.answers,
                "metadata": row.metadata,
                "similarity": row.similarity
            })
        
        return chunks
    
    def _generate_answer(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate an answer using the retrieved chunks.
        
        Args:
            question: The user's question.
            relevant_chunks: The retrieved relevant chunks.
            
        Returns:
            The generated answer.
        """
        # Create a prompt with the question and context from retrieved chunks
        context_text = "\n\n".join([chunk["text_chunk"] for chunk in relevant_chunks])
        
        prompt = f"""
        You are a helpful Google Cloud Platform expert assistant. Use the following context from Google Cloud documentation to answer the user's question accurately. Only use information from the provided context. If you don't know the answer or the context doesn't contain relevant information, say so instead of making up an answer.

        CONTEXT:
        {context_text}

        USER QUESTION:
        {question}
        
        ANSWER:
        """
        
        # Generate the answer
        response = self.generation_model.generate_content(prompt)
        return response.text

# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    import sys
    
    # Get environment variables or use defaults
    dataset = os.environ.get("BQ_DATASET", "your_dataset")
    table = os.environ.get("BQ_RAG_TABLE", "rag_chunks")
    
    if len(sys.argv) > 1:
        # Get question from command line arguments
        question = " ".join(sys.argv[1:])
        
        # Initialize the RAG system
        rag_system = RAGQuerySystem(bq_dataset=dataset, bq_rag_table=table)
        
        # Query the system
        result = rag_system.query(question)
        
        # Print the answer
        print(f"Question: {result['question']}")
        print("\nAnswer:")
        print(result['answer'])
        
        # Print the sources if requested
        if "--sources" in sys.argv:
            print("\nSources:")
            for i, chunk in enumerate(result['retrieved_chunks']):
                print(f"\nSource {i+1} (similarity: {chunk['similarity']:.4f}):")
                print(f"Document: {chunk['document_path']}")
                print(f"Category: {chunk['category']}")
                if 'metadata' in chunk and chunk['metadata']:
                    print(f"Metadata: {chunk['metadata']}")
    else:
        print("Please provide a question. Usage: python rag_query.py 'your question here'")