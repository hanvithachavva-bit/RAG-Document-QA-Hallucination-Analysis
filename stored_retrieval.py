import re
import faiss

from embeddings import model
from reranker import reranker


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_query(query):
    query = query.lower().strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    return query


def is_abstract_question(query):
    """Check whether the user explicitly asks about the abstract."""
    query = normalize_query(query)

    return "abstract" in query

def is_abstract_question(query):
    query = normalize_query(query)
    return "abstract" in query

def is_abstract_question(query):
    """Detect questions specifically asking about the abstract."""
    query = normalize_query(query)

    return any(phrase in query for phrase in (
        "abstract",
        "according to the abstract",
        "what does the abstract say",
        "what is stated in the abstract",
    ))

# ============================================================
# QUESTION TYPE
# ============================================================

def is_findings_question(query):
    """
    Detect questions asking about results, findings,
    performance, conclusions, or experimental outcomes.

    This is document-independent.
    """

    query = normalize_query(query)

    patterns = [
        r"\bmain findings\b",
        r"\bkey findings\b",
        r"\bmain results\b",
        r"\bkey results\b",
        r"\bfindings\b",
        r"\bresults\b",
        r"\bexperimental results\b",
        r"\bexperimental findings\b",
        r"\bexperiment show\b",
        r"\bwhat did.*experiment.*show\b",
        r"\bperformance\b",
        r"\bevaluation results\b",
        r"\bconclusion\b",
        r"\bconclusions\b"
    ]

    return any(
        re.search(pattern, query)
        for pattern in patterns
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_score(query, chunk):
    """
    Generic keyword overlap score.
    """

    query_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            query.lower()
        )
    )

    chunk_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            chunk.lower()
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
        "what",
        "which",
        "how",
        "why",
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
        "did",
        "do",
        "does"
    }

    query_words -= stop_words

    if not query_words:
        return 0.0

    overlap = query_words.intersection(
        chunk_words
    )

    return len(overlap) / len(query_words)


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = re.sub(
        r"\[SECTION:.*?\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# RESULT EVIDENCE SCORE
# ============================================================

def result_evidence_score(sentence):
    """
    Generic score for sentences that look like actual
    results/findings rather than experimental setup.

    No document-specific facts are used.
    """

    text = sentence.lower()

    score = 0.0

    # --------------------------------------------------------
    # Generic result terminology
    # --------------------------------------------------------

    result_terms = [
        "result",
        "results",
        "finding",
        "findings",
        "achieved",
        "performance",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "score",
        "scores",
        "improvement",
        "improved",
        "outperformed",
        "comparison",
        "compared",
        "evaluation",
        "evaluated",
        "demonstrated",
        "showed",
        "shows",
        "highest",
        "lowest",
        "better",
        "best",
        "significant",
        "significance",
        "statistically",
        "conclusion",
        "concludes",
        "metric",
        "metrics",
        "error",
        "errors"
    ]

    for term in result_terms:

        if term in text:
            score += 2.0

    # --------------------------------------------------------
    # Numerical evidence
    # --------------------------------------------------------

    numbers = re.findall(
        r"\b\d+(?:\.\d+)?%?\b",
        sentence
    )

    if numbers:

        score += min(
            len(numbers) * 1.5,
            6.0
        )

    # --------------------------------------------------------
    # Statistical evidence
    # --------------------------------------------------------

    if re.search(
        r"\bp\s*[<>=]\s*0?\.\d+",
        text
    ):
        score += 5.0

    if "confidence interval" in text:
        score += 3.0

    if "statistical significance" in text:
        score += 3.0

    # --------------------------------------------------------
    # Comparison language
    # --------------------------------------------------------

    comparison_terms = [
        "higher than",
        "lower than",
        "better than",
        "worse than",
        "greater than",
        "less than",
        "outperformed",
        "improvement over",
        "compared with",
        "compared to"
    ]

    for term in comparison_terms:

        if term in text:
            score += 3.0

    # --------------------------------------------------------
    # Penalize experimental setup
    # --------------------------------------------------------

    setup_terms = [
        "dataset was prepared",
        "dataset was created",
        "experimental setup",
        "configuration settings",
        "model parameters",
        "in this subsection",
        "subsections",
        "the dataset consists",
        "dataset consists",
        "were selected for fine-tuning",
        "selected for fine-tuning",
        "training configuration",
        "experimental setting",
        "experiment setting"
    ]

    for term in setup_terms:

        if term in text:
            score -= 4.0

    # --------------------------------------------------------
    # Penalize generic introductory sentences
    # --------------------------------------------------------

    generic_terms = [
        "this section describes",
        "this section presents",
        "this subsection describes",
        "details are given",
        "the following section",
        "the experiment involves"
    ]

    for term in generic_terms:

        if term in text:
            score -= 5.0

    return score


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search_chunks(
    query,
    document,
    top_k=20
):
    """
    Search the stored FAISS index using the query embedding.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = document["index"].search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        if index >= len(document["chunks"]):
            continue

        chunk = document["chunks"][index]

        results.append(
            (
                chunk,
                float(score)
            )
        )

    return results


# ============================================================
# COMBINE SEMANTIC + KEYWORD RESULTS
# ============================================================

def combine_candidates(
    query,
    semantic_results,
    chunks,
    max_candidates=30
):

    candidate_scores = {}

    # --------------------------------------------------------
    # Semantic candidates
    # --------------------------------------------------------

    for chunk, semantic_score in semantic_results:

        candidate_scores[chunk] = (
            candidate_scores.get(chunk, 0.0)
            + float(semantic_score) * 2.0
        )

    # --------------------------------------------------------
    # Keyword candidates
    # --------------------------------------------------------

    for chunk in chunks:

        keyword_score = calculate_keyword_score(
            query,
            chunk
        )

        if keyword_score > 0:

            candidate_scores[chunk] = (
                candidate_scores.get(chunk, 0.0)
                + keyword_score
            )

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:max_candidates]


# ============================================================
# RERANK USING CROSS ENCODER
# ============================================================

def rerank_candidates(
    query,
    candidates,
    top_k=20
):

    if not candidates:
        return []

    chunks = [
        chunk
        for chunk, score in candidates
    ]

    pairs = [
        [query, chunk]
        for chunk in chunks
    ]

    reranker_scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            chunks,
            reranker_scores
        ),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        (
            chunk,
            float(score)
        )
        for chunk, score in ranked[:top_k]
    ]


# ============================================================
# FINDINGS-SPECIFIC GENERIC RANKING
# ============================================================

def rank_findings_candidates(
    query,
    candidates,
    top_k=20
):
    """
    For findings/results questions, rank chunks based on
    the strongest evidence sentence inside each chunk.

    This improves retrieval without knowing anything about
    the actual document.
    """

    ranked_chunks = []

    for chunk, retrieval_score in candidates:

        sentences = split_sentences(
            chunk
        )

        if not sentences:
            continue

        best_sentence_score = max(
            result_evidence_score(sentence)
            for sentence in sentences
        )

        # Combine retrieval relevance with evidence strength
        final_score = (
            float(retrieval_score)
            + best_sentence_score
        )

        ranked_chunks.append(
            (
                chunk,
                final_score
            )
        )

    ranked_chunks.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_chunks[:top_k]


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_from_stored_document(
    query,
    document_id,
    top_k=20
):

    from stored_document import load_stored_document

    query = normalize_query(
        query
    )

    document = load_stored_document(
        document_id
    )

    chunks = document["chunks"]

    # For questions explicitly about the abstract,
    # retrieve abstract-labeled chunks directly.
    if is_abstract_question(query):
        abstract_chunks = [
            chunk
            for chunk in chunks
            if re.search(
                r"\[SECTION:\s*ABSTRACT\s*\]",
                chunk,
                flags=re.IGNORECASE,
            )
        ]

        if abstract_chunks:
            print(
                f"Found {len(abstract_chunks)} abstract chunks. "
                "Reranking abstract-specific evidence."
            )

            return rerank_candidates(
                query,
                [(chunk, 0.0) for chunk in abstract_chunks],
                top_k=min(top_k, len(abstract_chunks)),
            )

        

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    semantic_results = semantic_search_chunks(
        query,
        document,
        top_k=30
    )

    print(
        "\nSemantic candidates:",
        len(semantic_results)
    )

    # --------------------------------------------------------
    # Combine semantic + keyword retrieval
    # --------------------------------------------------------

    candidates = combine_candidates(
        query,
        semantic_results,
        chunks,
        max_candidates=30
    )

    print(
        "Combined candidates:",
        len(candidates)
    )
    
    # Prioritize abstract evidence for abstract-specific questions.
    if is_abstract_question(query):
        abstract_chunks = [
            (chunk, score)
            for chunk, score in candidates
            if re.search(
                r"\[SECTION:\s*ABSTRACT\s*\]",
                chunk,
                flags=re.IGNORECASE,
            )
        ]

        if abstract_chunks:
            print("Using abstract-specific evidence.")
            return rerank_candidates(
                query,
                abstract_chunks,
                top_k=top_k,
            )

    # --------------------------------------------------------
    # Findings/results questions
    # --------------------------------------------------------
    
    if is_findings_question(query):

        print(
            "Applying generic findings evidence ranking."
        )

        ranked = rank_findings_candidates(
            query,
            candidates,
            top_k=top_k
        )

        print(
            "\nTop findings evidence scores:"
        )

        for number, (chunk, score) in enumerate(
            ranked[:10],
            start=1
        ):

            print(
                number,
                "Score:",
                round(
                    float(score),
                    4
                )
            )

        return ranked

    # --------------------------------------------------------
    # Normal questions
    # --------------------------------------------------------

    print(
        "Applying cross-encoder reranking."
    )

    ranked = rerank_candidates(
        query,
        candidates,
        top_k=top_k
    )

    return ranked