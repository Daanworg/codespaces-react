# Simple entry point for Cloud Run
from rag_server import app

if __name__ == "__main__":
    # This allows you to run the app locally for development
    # In Cloud Run, the Gunicorn server will use the 'app' variable directly
    app.run(host="0.0.0.0", port=8080, debug=False)