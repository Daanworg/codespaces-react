# Fix main.py for Cloud Run
from rag_server import app

# This will be detected by Cloud Run's default Python 3 runtime
# No need to modify the service configuration