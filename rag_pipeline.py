from document_loader import extract_text_from_pdf, clean_text
from chunking import split_text
from embeddings import create_embeddings
from vector_store import create_vector_store
from retrieval import retrieve_chunks
from llm import generate_answer
from hallucination_analyzer import analyze_hallucination, calculate_metrics


def run_rag_pipeline(query):

    # ----------------------------------------
    # 1. Load and clean PDF
    # ----------------------------------------

    text = extract_text_from_pdf("sample.pdf")

    cleaned_text = clean_text(text)


    # ----------------------------------------
    # 2. Create chunks
    # ----------------------------------------

    chunks = split_text(cleaned_text)

    print("\nTotal chunks:", len(chunks))


    # ----------------------------------------
    # 3. Create embeddings
    # ----------------------------------------

    embeddings = create_embeddings(chunks)


    # ----------------------------------------
    # 4. Create FAISS index
    # ----------------------------------------

    index = create_vector_store(embeddings)

    print("Vectors in FAISS:", index.ntotal)


    # ----------------------------------------
    # 5. Retrieve relevant chunks
    # ----------------------------------------

    retrieved_chunks = retrieve_chunks(
        query,
        chunks,
        index,
        top_k=20
    )


    # ----------------------------------------
    # 6. Generate RAG answer
    # ----------------------------------------

    answer = generate_answer(
        query,
        retrieved_chunks
    )


    print("\n================================")
    print("RAG ANSWER")
    print("================================")

    print(answer)


    # ----------------------------------------
    # 7. Analyze hallucinations
    # ----------------------------------------

    analysis = analyze_hallucination(
        answer,
        retrieved_chunks
    )


    # ----------------------------------------
    # 8. Display claim analysis
    # ----------------------------------------

    print("\n================================")
    print("CLAIM ANALYSIS")
    print("================================")

    for i, result in enumerate(
        analysis,
        start=1
    ):

        print(f"\nClaim {i}:")
        print(result["claim"])

        print(
            "Classification:",
            result["classification"]
        )

        print(
            "Confidence:",
            round(
                result["confidence"],
                4
            )
        )


    # ----------------------------------------
    # 9. Calculate metrics
    # ----------------------------------------

    metrics = calculate_metrics(
        analysis
    )


    print("\n================================")
    print("HALLUCINATION METRICS")
    print("================================")

    print(
        "Supported:",
        round(metrics["supported"], 2),
        "%"
    )

    print(
        "Contradicted:",
        round(metrics["contradicted"], 2),
        "%"
    )

    print(
        "Unverifiable:",
        round(metrics["unverifiable"], 2),
        "%"
    )


    return {
        "answer": answer,
        "claims": analysis,
        "metrics": metrics
    }


if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    run_rag_pipeline(query)