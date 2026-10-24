import os
import json
import faiss


STORAGE_DIR = "data"
DOCUMENTS_DIR = os.path.join(STORAGE_DIR, "documents")
INDEXES_DIR = os.path.join(STORAGE_DIR, "indexes")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.json")


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_stored_document(document_id):

    metadata = load_metadata()

    if document_id not in metadata:
        raise ValueError(
            f"Document ID '{document_id}' was not found."
        )

    document_info = metadata[document_id]

    index_path = document_info["index_path"]
    chunks_path = document_info["chunks_path"]

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    index = faiss.read_index(index_path)

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    return {
        "document_id": document_info["document_id"],
        "document_name": document_info["document_name"],
        "file_hash": document_info["file_hash"],
        "chunk_count": document_info["chunk_count"],
        "index": index,
        "chunks": chunks
    }


if __name__ == "__main__":

    print("================================")
    print("STORED DOCUMENT TEST")
    print("================================")

    metadata = load_metadata()

    if not metadata:
        print("No stored documents found.")

    else:

        print(
            "Stored documents:",
            len(metadata)
        )

        for document_id in metadata:

            document = load_stored_document(
                document_id
            )

            print("\nDocument ID:")
            print(document["document_id"])

            print("Document name:")
            print(document["document_name"])

            print("Number of chunks:")
            print(document["chunk_count"])

            print("FAISS vectors:")
            print(document["index"].ntotal)

            print("Vector dimension:")
            print(document["index"].d)