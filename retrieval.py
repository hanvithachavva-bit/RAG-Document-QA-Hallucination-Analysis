import numpy as np

from document_loader import extract_text_from_pdf, clean_text
from chunking import split_text
from embeddings import create_embeddings
from vector_store import create_vector_store
from reranker import rerank_chunks


def detect_section(query):
    """
    Detect whether the query explicitly refers
    to a specific document section.
    """

    query_lower = query.lower()

    section_keywords = {
        "abstract": "ABSTRACT",
        "introduction": "I. INTRODUCTION",
        "literature review": "II. LITERATURE REVIEW",
        "decision making": "C. DECISION MAKING WITH LLM AGENT",
        "proposed methodology": "III. PROPOSED METHODOLOGY",
        "methodology": "III. PROPOSED METHODOLOGY",
        "unified grade": "C. UNIFIED GRADE AND SUMMARIZATION",
        "summarization": "C. UNIFIED GRADE AND SUMMARIZATION",
        "experiment and results": "IV. EXPERIMENT AND RESULTS",
        "model setting": "C. MODEL SETTING",
        "models": "C. MODEL SETTING",
        "conclusion": "V. CONCLUSION"
    }

    # Check longer phrases first
    # so that specific section names get priority.
    sorted_keywords = sorted(
        section_keywords.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for keyword, section_name in sorted_keywords:

        if keyword in query_lower:
            return section_name

    return None


def calculate_keyword_score(query, chunk):
    """
    Calculate keyword and phrase relevance.
    """

    query_lower = query.lower()
    chunk_lower = chunk.lower()

    score = 0

    # --------------------------------------------------
    # Important phrases
    # --------------------------------------------------

    phrases = [
        "sentence classification",
        "classification task",
        "models are selected",
        "selected for fine-tuning",
        "state-of-the-art models",
        "model setting",

        "resume screening",
        "screening process",
        "challenging process",

        "main findings",
        "main results",
        "key findings",
        "key results",
        "results and discussion",

        "evaluation and summarization",
        "resume evaluation",
        "resume summarization",

        "proposed system",
        "proposed methodology",
        "proposed approach"
    ]

    for phrase in phrases:

        if phrase in query_lower and phrase in chunk_lower:
            score += 5


    # --------------------------------------------------
    # Individual keywords
    # --------------------------------------------------

    keywords = [
        "model",
        "models",
        "classification",
        "sentence",
        "selected",
        "fine-tuning",
        "fine-tuned",

        "resume",
        "screening",
        "challenging",
        "applications",
        "candidates",

        "findings",
        "results",
        "discussion",
        "performance",
        "accuracy",
        "precision",
        "recall",
        "f1",

        "evaluation",
        "summarization",
        "summary",

        "methodology",
        "approach",
        "pipeline",
        "system"
    ]

    for keyword in keywords:

        if keyword in query_lower and keyword in chunk_lower:
            score += 1

    return score


def retrieve_chunks(query, chunks, index, top_k=20):

    target_section = detect_section(query)

    # ==================================================
    # CASE 1: SPECIFIC SECTION QUERY
    # ==================================================

    if target_section:

        section_chunks = []

        for chunk in chunks:

            if f"[SECTION: {target_section}]" in chunk:
                section_chunks.append(chunk)

        print(
            f"\nSection-aware retrieval: "
            f"{target_section}"
        )

        print(
            f"Chunks found in section: "
            f"{len(section_chunks)}"
        )

        # If the requested section does not exist,
        # fall back to normal document retrieval.
        if not section_chunks:

            print(
                "No chunks found in requested section."
            )

            target_section = None

        else:

            # ==================================================
            # STEP 1: SEMANTIC RETRIEVAL
            # ==================================================

            section_embeddings = create_embeddings(
                section_chunks
            )

            section_index = create_vector_store(
                section_embeddings
            )

            query_embedding = create_embeddings(
                [query]
            )

            query_vector = np.array(
                query_embedding
            ).astype("float32")

            semantic_top_k = min(
                10,
                len(section_chunks)
            )

            distances, indices = section_index.search(
                query_vector,
                semantic_top_k
            )

            semantic_candidates = []

            for i in indices[0]:

                if i < len(section_chunks):

                    semantic_candidates.append(
                        section_chunks[i]
                    )

            print(
                f"Semantic candidates: "
                f"{len(semantic_candidates)}"
            )


            # ==================================================
            # STEP 2: KEYWORD RETRIEVAL
            # ==================================================

            keyword_scored = []

            for chunk in section_chunks:

                score = calculate_keyword_score(
                    query,
                    chunk
                )

                keyword_scored.append(
                    (chunk, score)
                )

            keyword_scored.sort(
                key=lambda x: x[1],
                reverse=True
            )

            keyword_top_k = min(
                5,
                len(keyword_scored)
            )

            keyword_candidates = [
                chunk
                for chunk, score in keyword_scored[:keyword_top_k]
                if score > 0
            ]

            print(
                f"Keyword candidates: "
                f"{len(keyword_candidates)}"
            )


            # ==================================================
            # STEP 3: COMBINE CANDIDATES
            # ==================================================

            combined_candidates = []

            for chunk in semantic_candidates:

                if chunk not in combined_candidates:

                    combined_candidates.append(
                        chunk
                    )

            for chunk in keyword_candidates:

                if chunk not in combined_candidates:

                    combined_candidates.append(
                        chunk
                    )

            print(
                f"Combined candidates: "
                f"{len(combined_candidates)}"
            )


            # ==================================================
            # STEP 4: RERANK
            # ==================================================

            reranked_results = rerank_chunks(
                query,
                combined_candidates,
                top_k=3
            )

            return reranked_results


    # ==================================================
    # CASE 2: NORMAL DOCUMENT RETRIEVAL
    # ==================================================

    query_embedding = create_embeddings(
        [query]
    )

    query_vector = np.array(
        query_embedding
    ).astype("float32")


    # --------------------------------------------------
    # Semantic retrieval from main FAISS index
    # --------------------------------------------------

    distances, indices = index.search(
        query_vector,
        top_k
    )

    retrieved_chunks = []

    for i in indices[0]:

        if i < len(chunks):

            retrieved_chunks.append(
                chunks[i]
            )


    # --------------------------------------------------
    # Keyword retrieval across document
    # --------------------------------------------------

    keyword_scored = []

    for chunk in chunks:

        score = calculate_keyword_score(
            query,
            chunk
        )

        keyword_scored.append(
            (chunk, score)
        )

    keyword_scored.sort(
        key=lambda x: x[1],
        reverse=True
    )

    keyword_candidates = [
        chunk
        for chunk, score in keyword_scored[:5]
        if score > 0
    ]


    # --------------------------------------------------
    # Combine semantic + keyword candidates
    # --------------------------------------------------

    combined_candidates = []

    for chunk in retrieved_chunks:

        if chunk not in combined_candidates:

            combined_candidates.append(
                chunk
            )

    for chunk in keyword_candidates:

        if chunk not in combined_candidates:

            combined_candidates.append(
                chunk
            )


    # --------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------

    reranked_results = rerank_chunks(
        query,
        combined_candidates,
        top_k=3
    )

    return reranked_results


if __name__ == "__main__":

    # ==================================================
    # LOAD PDF
    # ==================================================

    text = extract_text_from_pdf(
        "sample.pdf"
    )

    # ==================================================
    # CLEAN TEXT
    # ==================================================

    cleaned_text = clean_text(
        text
    )

    # ==================================================
    # CREATE CHUNKS
    # ==================================================

    chunks = split_text(
        cleaned_text
    )

    print(
        "\nTotal chunks:",
        len(chunks)
    )

    # ==================================================
    # CREATE EMBEDDINGS
    # ==================================================

    embeddings = create_embeddings(
        chunks
    )

    # ==================================================
    # CREATE FAISS INDEX
    # ==================================================

    index = create_vector_store(
        embeddings
    )

    print(
        "Vectors in FAISS:",
        index.ntotal
    )

    # ==================================================
    # USER QUESTION
    # ==================================================

    query = input(
        "\nEnter your question: "
    )

    # ==================================================
    # DETECT SECTION
    # ==================================================

    detected_section = detect_section(
        query
    )

    print(
        "\nDetected section:",
        detected_section
    )

    # ==================================================
    # RETRIEVE
    # ==================================================

    results = retrieve_chunks(
        query,
        chunks,
        index,
        top_k=20
    )

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    print(
        "\nFinal Retrieved Chunks:\n"
    )

    for i, (chunk, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"--- CHUNK {i} ---"
        )

        print(
            "Reranker score:",
            score
        )

        print(
            chunk
        )

        print()