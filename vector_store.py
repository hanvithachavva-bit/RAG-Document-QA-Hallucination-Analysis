import faiss
import numpy as np


def create_vector_store(embeddings):
    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


if __name__ == "__main__":
    from chunking import split_text
    from document_loader import extract_text_from_pdf, clean_text
    from embeddings import create_embeddings

    text = extract_text_from_pdf("sample.pdf")

    cleaned_text = clean_text(text)

    chunks = split_text(cleaned_text)

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    print("Number of vectors in FAISS:", index.ntotal)
    print("Vector dimension:", index.d)