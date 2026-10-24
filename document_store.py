import os
import json
import hashlib

from document_loader import extract_text_from_pdf, clean_text
from chunking import split_text
from embeddings import create_embeddings
from vector_store import create_vector_store


STORAGE_DIR = "data"
DOCUMENTS_DIR = os.path.join(STORAGE_DIR, "documents")
INDEXES_DIR = os.path.join(STORAGE_DIR, "indexes")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.json")


def create_storage_directories():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def load_metadata():

    create_storage_directories()

    if not os.path.exists(METADATA_FILE):
        return {}

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_metadata(metadata):

    create_storage_directories()

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def get_document_id(file_path):

    file_hash = calculate_file_hash(file_path)

    # First 16 characters are enough for our local identifier.
    document_id = file_hash[:16]

    return document_id, file_hash


def document_already_exists(file_path):

    metadata = load_metadata()

    document_id, file_hash = get_document_id(file_path)

    if document_id in metadata:

        stored_hash = metadata[document_id]["file_hash"]

        if stored_hash == file_hash:
            return True, document_id

    return False, document_id


def process_and_store_document(file_path):

    create_storage_directories()

    already_exists, document_id = document_already_exists(
        file_path
    )

    if already_exists:

        print("\nDocument already exists.")

        print(
            "Reusing document ID:",
            document_id
        )

        return document_id

    print("\nProcessing new document...")

    print("Document ID:", document_id)

    text = extract_text_from_pdf(
        file_path
    )

    cleaned_text = clean_text(
        text
    )

    chunks = split_text(
        cleaned_text
    )

    print("Total chunks:", len(chunks))

    print("\nCreating embeddings...")

    embeddings = create_embeddings(
        chunks
    )

    print("Creating FAISS index...")

    index = create_vector_store(
        embeddings
    )

    # Save FAISS index.
    index_path = os.path.join(
        INDEXES_DIR,
        f"{document_id}.faiss"
    )

    import faiss

    faiss.write_index(
        index,
        index_path
    )

    # Save chunks.
    chunks_path = os.path.join(
        DOCUMENTS_DIR,
        f"{document_id}_chunks.json"
    )

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    metadata = load_metadata()

    file_hash = calculate_file_hash(
        file_path
    )

    metadata[document_id] = {
        "document_id": document_id,
        "document_name": os.path.basename(
            file_path
        ),
        "file_hash": file_hash,
        "chunk_count": len(chunks),
        "index_path": index_path,
        "chunks_path": chunks_path
    }

    save_metadata(
        metadata
    )

    print("\nDocument stored successfully.")

    print(
        "FAISS index:",
        index_path
    )

    print(
        "Chunks:",
        chunks_path
    )

    return document_id


if __name__ == "__main__":

    pdf_path = "sample.pdf"

    print("================================")
    print("DOCUMENT STORE TEST")
    print("================================")

    document_id = process_and_store_document(
        pdf_path
    )

    print("\nStored document ID:")
    print(document_id)