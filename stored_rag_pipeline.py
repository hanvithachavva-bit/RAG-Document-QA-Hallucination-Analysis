from document_manager import select_document
from stored_retrieval import retrieve_from_stored_document
from llm import generate_answer
from hallucination_analyzer import (
    analyze_hallucination,
    calculate_metrics
)


# ============================================================
# EVIDENCE RELEVANCE CHECK
# ============================================================
def evidence_is_relevant(
    query,
    retrieved_chunks,
    threshold=0.35
):
    if not retrieved_chunks:
        return False

    scores = []

    for chunk, score in retrieved_chunks[:5]:
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue

    if not scores:
        return False

    best_score = max(scores)

    print("\n================================")
    print("EVIDENCE RELEVANCE CHECK")
    print("================================")

    print(
        "Best reranker score:",
        round(best_score, 4)
    )

    print(
        "Required threshold:",
        threshold
    )

    query_lower = query.lower()

    # --------------------------------------------------------
    # SECTION-SPECIFIC QUESTIONS
    # --------------------------------------------------------

    section_keywords = [
        "abstract",
        "introduction",
        "literature review",
        "methodology",
        "proposed methodology",
        "model setting",
        "models",
        "summarization",
        "experiment and results",
        "conclusion"
    ]

    section_query = any(
        keyword in query_lower
        for keyword in section_keywords
    )

    if section_query:

        print("Query type: SECTION-SPECIFIC")

        # Section-specific questions must also contain
        # at least one meaningful word found in the evidence.

        query_words = set(
            word
            for word in query_lower.split()
            if len(word) >= 4
        )

        evidence_text = " ".join(
            chunk.lower()
            for chunk, _ in retrieved_chunks[:5]
        )

        matching_words = [
            word
            for word in query_words
            if word in evidence_text
        ]

        if matching_words:

            print(
                "Matching query terms:",
                matching_words
            )

            print(
                "Evidence status: RELEVANT"
            )

            return True

        print(
            "No meaningful query terms found in evidence."
        )

        print(
            "Evidence status: NOT RELEVANT"
        )

        return False

    # --------------------------------------------------------
    # GENERAL QUESTIONS
    # --------------------------------------------------------

    print("Query type: GENERAL")

    # Remove common question words.
    stop_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "about",
        "this",
        "that",
        "these",
        "those",
        "from",
        "used",
        "use",
        "main"
    }

    query_words = {
        word.strip(".,?!")
        for word in query_lower.split()
        if len(word.strip(".,?!")) >= 4
        and word.strip(".,?!") not in stop_words
    }

    evidence_text = " ".join(
        chunk.lower()
        for chunk, _ in retrieved_chunks[:5]
    )

    # --------------------------------------------------------
    # Small synonym groups for common document questions
    # --------------------------------------------------------

    synonym_groups = {
        "findings": {
            "findings",
            "results",
            "experiment",
            "performance",
            "accuracy",
            "metrics",
            "p-values"
        },

        "finding": {
            "finding",
            "results",
            "experiment",
            "performance",
            "accuracy",
            "metrics",
            "p-values"
        },

        "performance": {
            "performance",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "results",
            "metrics"
        },

        "models": {
            "models",
            "model",
            "llama2",
            "falcon",
            "flan-t5",
            "gpt-neox"
        },

        "methodology": {
            "methodology",
            "approach",
            "workflow",
            "process",
            "pipeline",
            "method"
        },

        "screening": {
            "screening",
            "resume",
            "candidate",
            "candidates",
            "applications"
        },

        "summarization": {
            "summarization",
            "summary",
            "summarize"
        }
    }

    expanded_query_terms = set(query_words)

    for word in query_words:

        if word in synonym_groups:

            expanded_query_terms.update(
                synonym_groups[word]
            )

    matching_words = [
        word
        for word in expanded_query_terms
        if word in evidence_text
    ]

    print(
        "Matching query terms:",
        matching_words
    )

    # --------------------------------------------------------
    # Relevance decision
    # --------------------------------------------------------

    # Require at least one meaningful connection between
    # the question and retrieved evidence.
    #
    # The reranker score alone is NOT sufficient.

    
    # The reranker score is a ranking signal, not a probability.
    # Require a meaningful query-to-evidence term match.
    if len(matching_words) >= 1:
        print("Evidence status: RELEVANT")
        return True

    print("Evidence status: NOT RELEVANT")
    return False


    

# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_stored_rag_pipeline():

    # --------------------------------------------------------
    # STEP 1: SELECT DOCUMENT
    # --------------------------------------------------------

    document = select_document()

    if document is None:

        print(
            "\nNo document selected."
        )

        return

    document_id = document["document_id"]


    # --------------------------------------------------------
    # STEP 2: GET USER QUESTION
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nQuestion cannot be empty."
        )

        return


    # --------------------------------------------------------
    # STEP 3: RETRIEVE INFORMATION
    # --------------------------------------------------------

    print("\n================================")
    print("RETRIEVING INFORMATION")
    print("================================")

    retrieved_chunks = retrieve_from_stored_document(
        query,
        document_id,
        top_k=20
    )


    # --------------------------------------------------------
    # STEP 4: CHECK EVIDENCE RELEVANCE
    # --------------------------------------------------------

    if not evidence_is_relevant(
        query,
        retrieved_chunks
    ):

        print("\n================================")
        print("NO SUFFICIENT EVIDENCE")
        print("================================")

        print(
            "The document does not contain enough "
            "relevant information to answer this question."
        )

        return


    # --------------------------------------------------------
    # STEP 5: GENERATE RAG ANSWER
    # --------------------------------------------------------

    print("\n================================")
    print("GENERATING ANSWER")
    print("================================")

    answer = generate_answer(
        query,
        retrieved_chunks
    )


    # --------------------------------------------------------
    # STEP 6: DISPLAY ANSWER
    # --------------------------------------------------------

    print("\n================================")
    print("RAG ANSWER")
    print("================================")

    print(answer)


    # --------------------------------------------------------
    # STEP 7: HALLUCINATION ANALYSIS
    # --------------------------------------------------------

    print("\n================================")
    print("CLAIM ANALYSIS")
    print("================================")

    analysis = analyze_hallucination(
        answer,
        retrieved_chunks
    )


    if not analysis:

        print(
            "No claims could be evaluated."
        )

    else:

        for i, result in enumerate(
            analysis,
            start=1
        ):

            print(
                f"\nClaim {i}:"
            )

            print(
                result["claim"]
            )

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


    # --------------------------------------------------------
    # STEP 8: CALCULATE HALLUCINATION METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        analysis
    )


    print("\n================================")
    print("HALLUCINATION METRICS")
    print("================================")

    print(
        "Supported:",
        round(
            metrics["supported"],
            2
        ),
        "%"
    )

    print(
        "Contradicted:",
        round(
            metrics["contradicted"],
            2
        ),
        "%"
    )

    print(
        "Unverifiable:",
        round(
            metrics["unverifiable"],
            2
        ),
        "%"
    )


    # --------------------------------------------------------
    # STEP 9: RETURN RESULTS
    # --------------------------------------------------------

    return {

        "document_id":
            document_id,

        "document_name":
            document["document_name"],

        "query":
            query,

        "answer":
            answer,

        "claims":
            analysis,

        "metrics":
            metrics
    }


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print("================================")
    print("STORED RAG PIPELINE")
    print("================================")

    run_stored_rag_pipeline()