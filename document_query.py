from document_manager import select_document
from stored_retrieval import retrieve_from_stored_document


def run_document_query():
    """
    Select a stored document and retrieve relevant
    chunks for a user question.
    """

    # Step 1: Select document
    document = select_document()

    if document is None:
        print("\nNo document selected.")
        return

    document_id = document["document_id"]

    # Step 2: Get user question
    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:
        print("\nQuestion cannot be empty.")
        return

    # Step 3: Retrieve relevant chunks
    print("\n================================")
    print("RETRIEVING INFORMATION")
    print("================================")

    results = retrieve_from_stored_document(
        query,
        document_id,
        top_k=20
    )

    # Step 4: Display retrieved chunks
    print("\n================================")
    print("FINAL RETRIEVED CHUNKS")
    print("================================")

    if not results:
        print("No relevant information found.")
        return

    for number, (chunk, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- CHUNK {number} ---"
        )

        print(
            "Reranker score:",
            round(float(score), 4)
        )

        print(chunk)


if __name__ == "__main__":

    print("================================")
    print("DOCUMENT QUERY SYSTEM")
    print("================================")

    run_document_query()