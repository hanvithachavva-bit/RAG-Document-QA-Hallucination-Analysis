from sentence_transformers import SentenceTransformer
from chunking import split_text
from document_loader import extract_text_from_pdf, clean_text


MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks):
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings


if __name__ == "__main__":
    text = extract_text_from_pdf("sample.pdf")

    cleaned_text = clean_text(text)

    chunks = split_text(cleaned_text)

    embeddings = create_embeddings(chunks)

    print("Number of embeddings:", len(embeddings))
    print("Embedding size:", len(embeddings[0]))