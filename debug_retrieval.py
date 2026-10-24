
import re

from document_manager import list_stored_documents
from stored_document import load_stored_document
from stored_retrieval import retrieve_from_stored_document
from llm import generate_answer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
QUERY = "What models were selected for the sentence classification task?"
DOCUMENT_NAME = "sample.pdf"


# --------------------------------------------------
# 1. FIND THE CORRECT DOCUMENT
# --------------------------------------------------

documents = list_stored_documents()

print("STORED DOCUMENTS:")
for doc in documents:
    print(doc["document_name"], doc["document_id"])

sample_document = next(
    (
        doc
        for doc in documents
        if doc["document_name"] == DOCUMENT_NAME
    ),
    None,
)

if sample_document is None:
    print(f"\nERROR: {DOCUMENT_NAME} was not found.")
    raise SystemExit(1)

document_id = sample_document["document_id"]

print(f"\nTesting document: {DOCUMENT_NAME}")
print("Document ID:", document_id)


# --------------------------------------------------
# 2. CHECK STORED ABSTRACT CHUNKS
# --------------------------------------------------

document = load_stored_document(document_id)
chunks = document["chunks"]

abstract_pattern = r"\[SECTION:\s*ABSTRACT\s*\]"

abstract_chunks = [
    chunk
    for chunk in chunks
    if re.search(
        abstract_pattern,
        chunk,
        flags=re.IGNORECASE,
    )
]

print("\n" + "=" * 60)
print("STORED DOCUMENT CHECK")
print("=" * 60)

print("Total stored chunks:", len(chunks))
print("Abstract-labeled chunks:", len(abstract_chunks))

for i, chunk in enumerate(abstract_chunks, start=1):
    print(f"\n--- Stored abstract chunk {i} ---")
    print(chunk[:1000])

if not abstract_chunks:
    print("WARNING: No abstract-labeled chunks were found.")


# --------------------------------------------------
# 3. RUN RETRIEVAL
# --------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVAL TEST")
print("=" * 60)

print("Question:", QUERY)

results = retrieve_from_stored_document(
    QUERY,
    document_id,
    top_k=5,
)

print("\nNumber of retrieved chunks:", len(results))

for i, item in enumerate(results, start=1):
    # Retrieval normally returns (chunk, score).
    chunk = item[0] if isinstance(item, tuple) else item
    score = item[1] if isinstance(item, tuple) else None

    print(f"\n--- Retrieved chunk {i} ---")

    if score is not None:
        print("Score:", round(float(score), 4))

    print(chunk[:1000])


# --------------------------------------------------
# 4. VERIFY ABSTRACT RETRIEVAL
# --------------------------------------------------

retrieved_abstract_count = sum(
    1
    for item in results
    if re.search(
        abstract_pattern,
        item[0] if isinstance(item, tuple) else item,
        flags=re.IGNORECASE,
    )
)

print("\n" + "=" * 60)
print("RETRIEVAL DIAGNOSTIC SUMMARY")
print("=" * 60)

print("Abstract chunks stored:", len(abstract_chunks))
print("Abstract chunks retrieved:", retrieved_abstract_count)

if not abstract_chunks:
    print("Result: Check PDF extraction or section splitting.")
elif retrieved_abstract_count == 0:
    print("Result: Abstract evidence was not retrieved.")
else:
    print("Result: Abstract evidence was retrieved successfully.")


# --------------------------------------------------
# 5. TEST ANSWER GENERATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANSWER GENERATION TEST")
print("=" * 60)

if not results:
    print("Cannot generate an answer because no chunks were retrieved.")
else:
    answer = generate_answer(QUERY, results)

    print("\nQUESTION:")
    print(QUERY)

    print("\nGENERATED ANSWER:")
    print(answer)
