from config import google_client, collection
from google.genai import errors


# query() is the core retrieval and answering function.
# It takes a question string, finds the most relevant chunks from ChromaDB,
# builds a context-grounded prompt, and returns an AI-generated answer.
def query(question: str):

    # Embed the user's question using the same Gemini Embedding model
    # that was used during ingestion. This converts the question into
    # a semantic number list so it can be compared against stored chunk embeddings.
    question_embedding = google_client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    # Search ChromaDB for the top 3 chunks whose embeddings are closest
    # to the question embedding. Closeness in embedding space means
    # similarity in meaning — not just keyword matching.
    results = collection.query(
        query_embeddings=question_embedding.embeddings[0].values,
        n_results=3
    )

    # results["documents"][0] returns a list of 3 chunk strings.
    # These are the most semantically relevant pieces of the ingested PDFs.
    relevant_chunks = results["documents"][0]

    # Combine the 3 chunks into one context string (~1500 words).
    # This context is what the AI will use to answer — it will not rely
    # on its own training data, only on what is provided here.
    context = " ".join(relevant_chunks)

    # Build the final prompt. The instruction "answer based on context only"
    # prevents the model from hallucinating answers outside the provided notes.
    prompt = f"Question: {question}\nAnswer based on the provided context only \n\nContext:\n{context}"

    # Ranked list of Gemini models to try in order.
    # Free tier accounts have per-model rate limits — if one model's quota
    # is exhausted, the next one is attempted automatically.
    models = [
        "models/gemini-3.1-pro-preview",
        "models/gemini-3.1-pro-preview-customtools",
        "models/gemini-3.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-pro-latest",
        "models/gemini-flash-latest",
        "models/gemini-3-flash-preview",
        "models/gemini-3.1-flash-lite",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash-lite-001",
        "models/gemini-flash-lite-latest",
        "models/gemma-4-31b-it",
        "models/gemma-4-26b-a4b-it"
    ]

    # Try each model in order. On a ClientError (rate limit or unavailable),
    # log the failure and move to the next model.
    # On success, return the answer text immediately and stop the loop.
    for model in models:
        try:
            response = google_client.models.generate_content(
                model=model,
                contents=prompt
            )
        except errors.ClientError as e:
            print(f"Model: {model} | Status: {e.status}")
        else:
            return response.text


# When query.py is run directly from the terminal, this block activates
# a simple CLI mode — prompts the user for a question and prints the answer.
# This block does not execute when query.py is imported by main.py.
if __name__ == "__main__":
    q = input("Ask a question: ")
    print(query(q))