from stored_document import load_metadata, load_stored_document


def list_stored_documents():
    """
    Return information about all documents stored in the system.
    """

    metadata = load_metadata()

    documents = []

    for document_id, info in metadata.items():

        documents.append({
            "document_id": document_id,
            "document_name": info["document_name"],
            "file_hash": info["file_hash"],
            "chunk_count": info["chunk_count"]
        })

    return documents


def display_documents():
    """
    Display all stored documents.
    """

    documents = list_stored_documents()

    print("\n================================")
    print("STORED DOCUMENTS")
    print("================================")

    if not documents:
        print("No documents found.")
        return

    for number, document in enumerate(documents, start=1):

        print(f"\nDocument {number}")
        print("----------------------------")

        print(
            "Document ID:",
            document["document_id"]
        )

        print(
            "Document name:",
            document["document_name"]
        )

        print(
            "Chunks:",
            document["chunk_count"]
        )


def get_document_by_id(document_id):
    """
    Load a stored document using its document ID.
    """

    documents = list_stored_documents()

    if document_id not in [doc["document_id"] for doc in documents]:

        print(
            "\nDocument ID not found:",
            document_id
        )

        return None

    return load_stored_document(document_id)


def select_document():
    """
    Display stored documents and allow the user
    to select one using its document ID.
    """

    documents = list_stored_documents()

    if not documents:

        print("\nNo stored documents available.")

        return None

    display_documents()

    document_id = input(
        "\nEnter the Document ID: "
    ).strip()

    document = get_document_by_id(document_id)

    if document is None:

        return None

    print("\n================================")
    print("DOCUMENT SELECTED")
    print("================================")

    print(
        "Document name:",
        document["document_name"]
    )

    print(
        "Document ID:",
        document["document_id"]
    )

    print(
        "Chunks:",
        document["chunk_count"]
    )

    print(
        "FAISS vectors:",
        document["index"].ntotal
    )

    return document


if __name__ == "__main__":

    document = select_document()

    if document is not None:

        print(
            "\nDocument loaded successfully."
        )