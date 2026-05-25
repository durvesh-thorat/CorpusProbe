from config import google_client, collection
from google.genai import errors


# query() is the core retrieval and answering function.
# It takes a question string, finds the most relevant chunks from ChromaDB,
# builds a context-grounded prompt, and returns an AI-generated answer.
def query(question: str):

    # Embed the question using the same model used during ingestion
    # so the vector space is consistent for similarity comparison.
    question_embedding = google_client.models.embed_content(
        model="gemini-embedding-2",
        contents=question
    )

    # Retrieve top 3 semantically similar chunks from ChromaDB.
    results = collection.query(
        query_embeddings=question_embedding.embeddings[0].values,
        n_results=3
    )

    relevant_chunks = results["documents"][0]
    context = " ".join(relevant_chunks)

    prompt = f"""You are a precise question-answering assistant. Your sole knowledge source is the context extracted from the user's personal PDF notes provided below.

        Rules:
        - Answer only using information present in the context.
        - If the answer is not in the context, respond with: "This information is not available in your notes."
        - Do not infer, assume, or use external knowledge.
        - Be concise and direct. Avoid unnecessary elaboration.
        - If the question is ambiguous, answer based on the most relevant part of the context.

        Context: {context}
        Question: {question}

        Answer:"""

    # Cascading fallback list — cycles to the next model on rate limit errors.
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