import os
import re

import streamlit as st

from document_store import process_and_store_document
from document_manager import list_stored_documents
from stored_retrieval import retrieve_from_stored_document
from stored_rag_pipeline import evidence_is_relevant
from llm import generate_answer
from hallucination_analyzer import (
    analyze_hallucination,
    calculate_metrics
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# HELPER — CLEAN EVIDENCE
# ============================================================

def clean_evidence_text(evidence):
    """
    Remove internal formatting from retrieved evidence.
    """

    if not evidence:
        return ""

    # Remove section markers such as:
    # [SECTION: IV. EXPERIMENT AND RESULTS]
    cleaned = re.sub(
        r"\[SECTION:.*?\]\s*",
        "",
        evidence,
        flags=re.IGNORECASE
    )

    # Remove citation numbers such as [15] or [15, 16]
    cleaned = re.sub(
        r"\s*\[\d+(?:,\s*\d+)*\]",
        "",
        cleaned
    )

    # Normalize whitespace
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    )

    return cleaned.strip()


# ============================================================
# HELPER — FIND RELEVANT EVIDENCE FOR A CLAIM
# ============================================================

def get_relevant_evidence(claim, evidence):
    """
    Select the most relevant sentence(s) from the evidence
    for displaying alongside a hallucination claim.

    This function is completely document-independent.
    """

    if not evidence:
        return None

    cleaned = clean_evidence_text(
        evidence
    )

    if not cleaned:
        return None

    # --------------------------------------------------------
    # Split evidence into sentences
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return cleaned[:800]

    # --------------------------------------------------------
    # Extract important claim words
    # --------------------------------------------------------

    claim_words = set(
        re.findall(
            r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
            claim.lower()
        )
    )

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "them",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "by",
        "from",
        "as",
        "which",
        "than"
    }

    claim_words -= stop_words

    # --------------------------------------------------------
    # Score each sentence by lexical overlap
    # --------------------------------------------------------

    candidates = []

    for sentence in sentences:

        sentence_words = set(
            re.findall(
                r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
                sentence.lower()
            )
        )

        sentence_words -= stop_words

        overlap = len(
            claim_words.intersection(
                sentence_words
            )
        )

        # Numerical values are important evidence
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            sentence
        )

        score = float(overlap)

        if numbers:
            score += min(
                len(numbers) * 0.5,
                2.0
            )

        if score > 0:
            candidates.append(
                (
                    sentence,
                    score
                )
            )

    # --------------------------------------------------------
    # Return strongest evidence
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    if candidates:

        selected = []

        for sentence, score in candidates:

            if sentence not in selected:

                selected.append(
                    sentence
                )

            if len(selected) >= 2:
                break

        return " ".join(
            selected
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(cleaned) > 800:
        return cleaned[:800] + "..."

    return cleaned


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 RAG-Based Document Q&A System"
)

st.write(
    "Upload a document, select a stored document, "
    "ask a question, and analyze the generated answer "
    "for hallucinations."
)


# ============================================================
# SIDEBAR — DOCUMENT MANAGEMENT
# ============================================================

st.sidebar.header(
    "Document Management"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS UPLOADED DOCUMENT
# ============================================================

if uploaded_file is not None:

    os.makedirs(
        "uploaded_documents",
        exist_ok=True
    )

    uploaded_path = os.path.join(
        "uploaded_documents",
        uploaded_file.name
    )

    with open(
        uploaded_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    if st.sidebar.button(
        "Process Document"
    ):

        with st.spinner(
            "Processing document..."
        ):

            document_id = (
                process_and_store_document(
                    uploaded_path
                )
            )

        st.sidebar.success(
            "Document processed successfully."
        )

        st.sidebar.write(
            "Document ID:",
            document_id
        )


# ============================================================
# LOAD STORED DOCUMENTS
# ============================================================

documents = list_stored_documents()


if not documents:

    st.warning(
        "No documents are currently stored. "
        "Upload and process a PDF to begin."
    )

    st.stop()


# ============================================================
# DOCUMENT SELECTION
# ============================================================

st.subheader(
    "Select Document"
)

document_options = {
    document["document_name"]:
    document["document_id"]
    for document in documents
}

selected_document_name = st.selectbox(
    "Document",
    list(
        document_options.keys()
    )
)

selected_document_id = (
    document_options[
        selected_document_name
    ]
)


# ============================================================
# SELECTED DOCUMENT INFORMATION
# ============================================================

selected_document = next(
    document
    for document in documents
    if document["document_id"]
    == selected_document_id
)


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Document",
        selected_document[
            "document_name"
        ]
    )


with col2:

    st.metric(
        "Chunks",
        selected_document[
            "chunk_count"
        ]
    )


with col3:

    st.metric(
        "Document ID",
        selected_document[
            "document_id"
        ][:12] + "..."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "Ask a Question"
)

query = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What are the main findings "
        "of the experiment?"
    )
)

ask_button = st.button(
    "🔍 Ask Question",
    type="primary"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Retrieving relevant information..."
    ):

        retrieved_chunks = (
            retrieve_from_stored_document(
                query,
                selected_document_id,
                top_k=20
            )
        )
        for i, item in enumerate(retrieved_chunks[:5], start=1):
            chunk = item[0] if isinstance(item, tuple) else item
            print(f"\n--- Retrieved chunk {i} ---\n{chunk[:700]}")

        if not retrieved_chunks:

            st.error(
                "No relevant information was found "
                "in the selected document."
            )

            st.stop()


    # --------------------------------------------------------
    # CHECK WHETHER EVIDENCE IS ACTUALLY RELEVANT
    # --------------------------------------------------------

        evidence_relevant = evidence_is_relevant(
            query,
            retrieved_chunks
        )


        if not evidence_relevant:

            st.warning(
                "The selected document does not contain "
                "enough relevant information to answer this question."
            )

            st.subheader("Answer")

            st.write(
                "The document does not contain enough relevant "
                "information to answer this question."
            )

            st.subheader("Hallucination Analysis")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Supported",
                    "0.00%"
                )

            with metric2:
                st.metric(
                    "Contradicted",
                    "0.00%"
                )

            with metric3:
                st.metric(
                    "Unverifiable",
                    "0.00%"
                )

            st.subheader("Claim-Level Analysis")

            st.info(
                "No claims were generated because the question "
                "was not relevant to the selected document."
            )

            st.stop()


    # --------------------------------------------------------
    # ANSWER GENERATION
    # --------------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                query,
                retrieved_chunks
            )

    # --------------------------------------------------------
    # CLEAN FINAL ANSWER
    # --------------------------------------------------------

    answer = answer.strip()

    # Remove common model generation artifacts
    answer = answer.replace(
        "[END_OF_TEXT]",
        ""
    )

    answer = answer.replace(
        "[END OF TEXT]",
        ""
    )

    answer = answer.replace(
        "<|endoftext|>",
        ""
    )

    answer = answer.strip()

    # Remove accidental incomplete trailing phrase
    answer = re.sub(
        r"\bTo determine if the given text contains "
        r"relevant information about.*$",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    st.subheader(
        "Answer"
    )

    st.write(
        answer
    )

    # --------------------------------------------------------
    # HALLUCINATION ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing answer..."
    ):

        analysis = analyze_hallucination(
            answer,
            retrieved_chunks
        )

        metrics = calculate_metrics(
            analysis
        )

    # ========================================================
    # HALLUCINATION METRICS
    # ========================================================

    st.subheader(
        "Hallucination Analysis"
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    with metric1:

        st.metric(
            "Supported",
            f"{metrics['supported']:.2f}%"
        )

    with metric2:

        st.metric(
            "Contradicted",
            f"{metrics['contradicted']:.2f}%"
        )

    with metric3:

        st.metric(
            "Unverifiable",
            f"{metrics['unverifiable']:.2f}%"
        )

    # ========================================================
    # CLAIM-LEVEL ANALYSIS
    # ========================================================

    st.subheader(
        "Claim-Level Analysis"
    )

    if not analysis:

        st.info(
            "No claims could be evaluated."
        )

    else:

        for number, result in enumerate(
            analysis,
            start=1
        ):

            classification = (
                result["classification"]
            )

            # ------------------------------------------------
            # Classification icon
            # ------------------------------------------------

            if classification == "Supported":

                icon = "✅"

            elif classification == "Contradicted":

                icon = "❌"

            else:

                icon = "⚠️"

            # ------------------------------------------------
            # Claim expander
            # ------------------------------------------------

            with st.expander(
                f"{icon} Claim {number}: "
                f"{classification}"
            ):

                st.write(
                    "**Claim:**"
                )

                st.write(
                    result["claim"]
                )

                st.write(
                    "**Confidence:**",
                    round(
                        result["confidence"],
                        4
                    )
                )

                # ------------------------------------------------
                # Focused evidence
                # ------------------------------------------------

                if result["evidence"]:

                    focused_evidence = (
                        get_relevant_evidence(
                            result["claim"],
                            result["evidence"]
                        )
                    )

                    if focused_evidence:

                        st.write(
                            "**Evidence:**"
                        )

                        st.info(
                            focused_evidence
                        )