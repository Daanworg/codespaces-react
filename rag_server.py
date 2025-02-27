# A simple Flask server to provide a web interface for the RAG query system
import os
from flask import Flask, request, jsonify, render_template_string
from rag_query import RAGQuerySystem

app = Flask(__name__)

# Get environment variables
dataset = os.environ.get("BQ_DATASET", "your_dataset")
table = os.environ.get("BQ_RAG_TABLE", "rag_chunks")

# Initialize the RAG system
rag_system = RAGQuerySystem(bq_dataset=dataset, bq_rag_table=table)

# Simple HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Google Cloud Documentation RAG System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .query-box {
            width: 100%;
            padding: 20px;
            box-sizing: border-box;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .query-input {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
        .query-button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4285f4;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .answer-box {
            padding: 20px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .source-box {
            padding: 20px;
            border-radius: 4px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }
        .source-meta {
            margin-bottom: 10px;
            font-size: 14px;
            color: #666;
        }
        .loading {
            text-align: center;
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Google Cloud Documentation RAG System</h1>
        <p>Ask questions about Google Cloud Platform</p>
    </div>
    
    <div class="query-box">
        <input type="text" id="queryInput" class="query-input" placeholder="Ask a question about Google Cloud...">
        <button onclick="submitQuery()" class="query-button">Ask</button>
        <div class="loading" id="loading">
            <p>Searching and generating answer...</p>
        </div>
    </div>
    
    <div id="answerContainer" style="display: none;">
        <h2>Answer</h2>
        <div class="answer-box" id="answerBox"></div>
        
        <h2>Sources</h2>
        <div id="sourcesContainer"></div>
    </div>
    
    <script>
        function submitQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query) return;
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('answerContainer').style.display = 'none';
            
            // Submit query to API
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answerContainer').style.display = 'block';
                
                // Show answer
                document.getElementById('answerBox').innerText = data.answer;
                
                // Show sources
                const sourcesContainer = document.getElementById('sourcesContainer');
                sourcesContainer.innerHTML = '';
                
                data.retrieved_chunks.forEach((chunk, index) => {
                    const sourceBox = document.createElement('div');
                    sourceBox.className = 'source-box';
                    
                    const sourceMeta = document.createElement('div');
                    sourceMeta.className = 'source-meta';
                    sourceMeta.innerHTML = `<strong>Source ${index + 1}</strong> | Similarity: ${chunk.similarity.toFixed(4)} | Category: ${chunk.category}`;
                    
                    const sourceText = document.createElement('div');
                    sourceText.innerText = chunk.text_chunk;
                    
                    sourceBox.appendChild(sourceMeta);
                    sourceBox.appendChild(sourceText);
                    sourcesContainer.appendChild(sourceBox);
                });
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        }
        
        // Allow submitting by pressing Enter
        document.getElementById('queryInput').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                submitQuery();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
def query():
    """API endpoint for querying the RAG system."""
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = rag_system.query(query_text)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For development only - not for production use
    app.run(host='0.0.0.0', port=8080, debug=True)