import google.genai as genai
import chromadb
import os
from dotenv import load_dotenv

# Load environment variables from the .env file into os.environ.
# This must be called before os.getenv() — otherwise the API key returns None.
load_dotenv()

# Create a persistent ChromaDB client that stores data in the ./db directory.
# PersistentClient ensures embeddings and documents survive across sessions —
# unlike an in-memory client which resets every time the program exits.
db_client = chromadb.PersistentClient(path="./db")

# Get or create a collection named "sources" inside ChromaDB.
# A collection is equivalent to a table — it holds all ingested chunks
# along with their embeddings and metadata.
collection = db_client.get_or_create_collection(name="sources")

# Initialize the Gemini client using the API key stored in .env.
# This single client instance is shared across ingest.py, query.py,
# and main.py by importing from this file — avoiding repeated initialization.
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))