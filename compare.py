from document_manager import get_document_by_id
from stored_retrieval import retrieve_from_stored_document

from llm import generate_answer
from baseline import generate_non_rag_answer

from hallucination_analyzer import (
    analyze_hallucination,
    calculate_metrics
)

from stored_rag_pipeline import evidence_is_relevant
from answer_evaluator import evaluate_answer_quality


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

evaluation_questions = [
    "What does the abstract say about the proposed system?",
    "Why is resume screening considered a challenging process?",
    "What is the main approach used in the proposed methodology?",
    "How are resume evaluation and summarization performed?",
    "What models are selected for the sentence classification task?",
    "What are the main findings of the experiment?"
]


# ============================================================
# DOCUMENT
# ============================================================

DOCUMENT_ID = "93b8f443f7de325b"


# ============================================================
# RUN COMPARISON
# ============================================================

def run_comparison():

    print("\n================================")
    print("LOADING STORED DOCUMENT")
    print("================================")

    document = get_document_by_id(
        DOCUMENT_ID
    )

    if document is None:

        print(
            "\nDocument could not be loaded."
        )

        return

    print(
        "Document:",
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


    comparison_results = []


    # ========================================================
    # RUN EACH QUESTION
    # ========================================================

    for number, question in enumerate(
        evaluation_questions,
        start=1
    ):

        print("\n\n================================")
        print(
            f"QUESTION {number}"
        )
        print("================================")

        print(question)


        # ====================================================
        # RETRIEVE EVIDENCE
        # ====================================================

        print("\n================================")
        print("RETRIEVING EVIDENCE")
        print("================================")

        retrieved_chunks = retrieve_from_stored_document(
            question,
            DOCUMENT_ID,
            top_k=20
        )


        # ====================================================
        # RAG
        # ====================================================

        print("\n---------- RAG ----------")


        rag_relevant = evidence_is_relevant(
            question,
            retrieved_chunks
        )


        if rag_relevant:

            rag_answer = generate_answer(
                question,
                retrieved_chunks
            )

        else:

            rag_answer = (
                "The document does not contain enough "
                "relevant information to answer this question."
            )


        print("\nRAG ANSWER:")
        print(rag_answer)


        # ----------------------------------------------------
        # RAG ANSWER QUALITY
        # ----------------------------------------------------

        rag_quality = evaluate_answer_quality(
            rag_answer
        )

        print(
            "RAG ANSWER QUALITY:",
            rag_quality
        )


        # ----------------------------------------------------
        # RAG HALLUCINATION ANALYSIS
        # ----------------------------------------------------

        rag_analysis = analyze_hallucination(
            rag_answer,
            retrieved_chunks
        )
        print("\nRAG CLAIM-LEVEL DEBUG:")
        for item in rag_analysis:
            print("Claim:", item["claim"])
            print("Classification:", item["classification"])
            print("Confidence:", item["confidence"])
            print("Evidence:", item["evidence"])
            print("-" * 50)

        rag_metrics = calculate_metrics(
            rag_analysis
        )


        print("\nRAG METRICS:")

        print(
            "Supported:",
            round(
                rag_metrics["supported"],
                2
            ),
            "%"
        )

        print(
            "Contradicted:",
            round(
                rag_metrics["contradicted"],
                2
            ),
            "%"
        )

        print(
            "Unverifiable:",
            round(
                rag_metrics["unverifiable"],
                2
            ),
            "%"
        )


        # ====================================================
        # NON-RAG
        # ====================================================

        print("\n---------- NON-RAG ----------")


        # IMPORTANT:
        # The Non-RAG model receives ONLY the question.
        # It does NOT receive the PDF or retrieved chunks.

        non_rag_answer = generate_non_rag_answer(
            question
        )


        print("\nNON-RAG ANSWER:")
        print(non_rag_answer)


        # ----------------------------------------------------
        # NON-RAG ANSWER QUALITY
        # ----------------------------------------------------

        non_rag_quality = evaluate_answer_quality(
            non_rag_answer
        )

        print(
            "NON-RAG ANSWER QUALITY:",
            non_rag_quality
        )


        # ----------------------------------------------------
        # NON-RAG HALLUCINATION ANALYSIS
        # ----------------------------------------------------
        #
        # The retrieved document evidence is used ONLY
        # after generation to evaluate the baseline.
        #
        # The baseline model itself never sees this evidence.
        # ----------------------------------------------------

        non_rag_analysis = analyze_hallucination(
            non_rag_answer,
            retrieved_chunks
        )

        non_rag_metrics = calculate_metrics(
            non_rag_analysis
        )


        print("\nNON-RAG METRICS:")

        print(
            "Supported:",
            round(
                non_rag_metrics["supported"],
                2
            ),
            "%"
        )

        print(
            "Contradicted:",
            round(
                non_rag_metrics["contradicted"],
                2
            ),
            "%"
        )

        print(
            "Unverifiable:",
            round(
                non_rag_metrics["unverifiable"],
                2
            ),
            "%"
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        comparison_results.append({

            "question":
                question,

            "rag":
                rag_metrics,

            "rag_quality":
                rag_quality,

            "non_rag":
                non_rag_metrics,

            "non_rag_quality":
                non_rag_quality

        })


    # ========================================================
    # ANSWER QUALITY SUMMARY
    # ========================================================

    print("\n\n================================")
    print("ANSWER QUALITY SUMMARY")
    print("================================")


    quality_categories = [
        "Good",
        "Incomplete",
        "Poor",
        "No Answer"
    ]


    for category in quality_categories:

        rag_count = sum(
            1
            for result in comparison_results
            if result["rag_quality"] == category
        )

        non_rag_count = sum(
            1
            for result in comparison_results
            if result["non_rag_quality"] == category
        )


        print(
            f"{category}: "
            f"RAG = {rag_count}, "
            f"Non-RAG = {non_rag_count}"
        )


    # ========================================================
    # OVERALL RAG VS NON-RAG
    # ========================================================

    print("\n\n================================")
    print("OVERALL RAG VS NON-RAG")
    print("================================")


    total = len(
        comparison_results
    )


    if total == 0:

        print(
            "No comparison results available."
        )

        return


    # --------------------------------------------------------
    # RAG averages
    # --------------------------------------------------------

    rag_supported = sum(
        result["rag"]["supported"]
        for result in comparison_results
    ) / total


    rag_contradicted = sum(
        result["rag"]["contradicted"]
        for result in comparison_results
    ) / total


    rag_unverifiable = sum(
        result["rag"]["unverifiable"]
        for result in comparison_results
    ) / total


    # --------------------------------------------------------
    # Non-RAG averages
    # --------------------------------------------------------

    non_rag_supported = sum(
        result["non_rag"]["supported"]
        for result in comparison_results
    ) / total


    non_rag_contradicted = sum(
        result["non_rag"]["contradicted"]
        for result in comparison_results
    ) / total


    non_rag_unverifiable = sum(
        result["non_rag"]["unverifiable"]
        for result in comparison_results
    ) / total


    # ========================================================
    # PRINT RAG RESULTS
    # ========================================================

    print("\nRAG:")

    print(
        "Average Supported:",
        round(
            rag_supported,
            2
        ),
        "%"
    )

    print(
        "Average Contradicted:",
        round(
            rag_contradicted,
            2
        ),
        "%"
    )

    print(
        "Average Unverifiable:",
        round(
            rag_unverifiable,
            2
        ),
        "%"
    )


    # ========================================================
    # PRINT NON-RAG RESULTS
    # ========================================================

    print("\nNON-RAG:")

    print(
        "Average Supported:",
        round(
            non_rag_supported,
            2
        ),
        "%"
    )

    print(
        "Average Contradicted:",
        round(
            non_rag_contradicted,
            2
        ),
        "%"
    )

    print(
        "Average Unverifiable:",
        round(
            non_rag_unverifiable,
            2
        ),
        "%"
    )


if __name__ == "__main__":

    run_comparison()