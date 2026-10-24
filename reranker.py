from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank_chunks(query, chunks, top_k=3):
    pairs = []

    for chunk in chunks:
        pairs.append([query, chunk])

    scores = reranker.predict(pairs)

    ranked_chunks = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_chunks[:top_k]


if __name__ == "__main__":
    query = "What does the abstract say about the proposed system?"

    chunks = [
        "The proposed system uses LLMs for automated resume screening and grading.",
        "The paper discusses statistical tests and confidence intervals.",
        "The system performs candidate evaluation using grades and summaries."
    ]

    results = rerank_chunks(query, chunks)

    for chunk, score in results:
        print("\nScore:", score)
        print(chunk)