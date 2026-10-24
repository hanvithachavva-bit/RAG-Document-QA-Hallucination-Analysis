from document_manager import get_document_by_id
from stored_retrieval import retrieve_from_stored_document

from llm import generate_answer

from hallucination_analyzer import (
    analyze_hallucination,
    calculate_metrics
)

from stored_rag_pipeline import evidence_is_relevant


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
# DOCUMENT TO EVALUATE
# ============================================================

DOCUMENT_ID = "93b8f443f7de325b"


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

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


    evaluation_results = []


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


        # ----------------------------------------------------
        # RETRIEVE USING THE ACTUAL STORED RAG PIPELINE
        # ----------------------------------------------------

        print("\n================================")
        print("RETRIEVING INFORMATION")
        print("================================")

        retrieved_chunks = retrieve_from_stored_document(
            question,
            DOCUMENT_ID,
            top_k=20
        )


        # ----------------------------------------------------
        # RELEVANCE CHECK
        # ----------------------------------------------------

        if not evidence_is_relevant(
            question,
            retrieved_chunks
        ):

            print("\n================================")
            print("NO SUFFICIENT EVIDENCE")
            print("================================")

            answer = (
                "The document does not contain enough "
                "relevant information to answer this question."
            )

            print(answer)

            metrics = {
                "supported": 0,
                "contradicted": 0,
                "unverifiable": 0
            }

            evaluation_results.append({

                "question": question,

                "answer": answer,

                "metrics": metrics

            })

            continue


        # ----------------------------------------------------
        # GENERATE RAG ANSWER
        # ----------------------------------------------------

        print("\n================================")
        print("GENERATING ANSWER")
        print("================================")

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        print("\nRAG ANSWER:")
        print(answer)


        # ----------------------------------------------------
        # HALLUCINATION ANALYSIS
        # ----------------------------------------------------

        print("\n================================")
        print("HALLUCINATION ANALYSIS")
        print("================================")

        analysis = analyze_hallucination(
            answer,
            retrieved_chunks
        )

        metrics = calculate_metrics(
            analysis
        )


        print("\nHALLUCINATION METRICS:")

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


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        evaluation_results.append({

            "question": question,

            "answer": answer,

            "metrics": metrics

        })


    # ========================================================
    # OVERALL EVALUATION
    # ========================================================

    print("\n\n================================")
    print("OVERALL EVALUATION")
    print("================================")


    total_questions = len(
        evaluation_results
    )


    if total_questions == 0:

        print(
            "No questions were evaluated."
        )

        return


    total_supported = sum(
        result["metrics"]["supported"]
        for result in evaluation_results
    )

    total_contradicted = sum(
        result["metrics"]["contradicted"]
        for result in evaluation_results
    )

    total_unverifiable = sum(
        result["metrics"]["unverifiable"]
        for result in evaluation_results
    )


    print(
        "Average Supported:",
        round(
            total_supported / total_questions,
            2
        ),
        "%"
    )

    print(
        "Average Contradicted:",
        round(
            total_contradicted / total_questions,
            2
        ),
        "%"
    )

    print(
        "Average Unverifiable:",
        round(
            total_unverifiable / total_questions,
            2
        ),
        "%"
    )


if __name__ == "__main__":

    run_evaluation()