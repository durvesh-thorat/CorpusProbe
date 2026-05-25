from config import collection, os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ingest import ingest
from query import query


# Query is a Pydantic model that defines the expected shape of the POST /query
# request body. FastAPI uses this to automatically validate incoming JSON —
# if "question" is missing or not a string, it returns a 422 error instantly.
class Query(BaseModel):
    question: str


app = FastAPI(title="CORPUS PROBE")


# Returns a set of unique filenames that have been ingested into ChromaDB.
# collection.get() fetches all stored records — we extract just the filename
# from each record's metadata and deduplicate using set().
@app.get("/sources")
def get_sources():
    result = collection.get(include=["metadatas"])
    files = [m["filename"] for m in result["metadatas"]]
    return set(files)


# Accepts a PDF file upload, saves it to the sources/ directory,
# then runs the full ingestion pipeline — chunking, embedding, and
# storing into ChromaDB — so it becomes queryable immediately.
# async def is used because file.read() is an async I/O operation.
@app.post("/sources")
async def upload_source(file: UploadFile = File(...)):

    # Save uploaded file to disk in binary mode
    contents = await file.read()
    with open(f"sources/{file.filename}", "wb") as f:
        f.write(contents)

    ingest(f"sources/{file.filename}")

    return {"message": f"{file.filename} ingested successfully"}


# Accepts a question via JSON request body, passes it to query()
# which retrieves relevant chunks from ChromaDB and generates
# a context-grounded answer using the Gemini model.
@app.post("/query")
def get_query(q: Query):
    return query(q.question)


# Removes a source from both ChromaDB and the sources/ directory.
# collection.delete(where=...) deletes all chunks whose metadata
# filename matches — this handles multi-chunk PDFs cleanly.
# os.remove() then deletes the actual PDF file from disk.
@app.delete("/sources/{filename}")
def delete_source(filename: str):
    collection.delete(where={"filename": filename})
    os.remove(f"sources/{filename}")
    return f"File: {filename} has been deleted successfully!"