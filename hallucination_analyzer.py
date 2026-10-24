import re

from claim_extractor import extract_claims
from nli_classifier import classify_claim
from reranker import reranker


# ----------------------------- Text utilities -----------------------------
def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)
    text = re.sub(r"\[SECTION:.*?\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def clean_claims(claims):
    cleaned, seen = [], set()
    for claim in claims or []:
        claim = normalize_text(str(claim)).strip(" \t\n-•")
        if len(claim.split()) < 5:
            continue
        key = claim.casefold()
        if key not in seen:
            cleaned.append(claim)
            seen.add(key)
    return cleaned


def split_sentences(text):
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [s.strip() for s in parts if len(s.strip().split()) >= 5]


# ----------------------------- Answer quality -----------------------------
def is_poor_answer(answer):
    answer = (answer or "").strip()
    if not answer:
        return True
    words = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", answer.lower())
    if len(words) < 5:
        return True
    single_letters = sum(len(word) == 1 for word in words)
    if len(words) >= 8 and single_letters / len(words) >= 0.5:
        return True
    if len(words) >= 12:
        for size in range(3, 6):
            counts = {}
            for i in range(len(words) - size + 1):
                phrase = tuple(words[i:i + size])
                counts[phrase] = counts.get(phrase, 0) + 1
            if any(count >= 4 for count in counts.values()):
                return True
    return False


# ----------------------------- Lexical overlap ----------------------------
_STOP_WORDS = {
    "the", "and", "are", "is", "was", "were", "for", "of", "to", "a",
    "an", "in", "on", "with", "this", "that", "these", "those", "as",
    "by", "from", "it", "their", "they", "has", "have", "had", "also",
    "than", "which", "its", "be", "been", "being", "did", "does", "do",
    "can", "could", "would", "should", "may", "might", "will", "not",
    "into", "over", "under", "such", "while", "when", "where", "who",
    "what", "how", "because", "due", "there", "here", "then",
}


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower()))


def content_word_overlap(claim, evidence):
    claim_words = tokenize(claim) - _STOP_WORDS
    evidence_words = tokenize(evidence) - _STOP_WORDS
    if not claim_words:
        return 0.0
    return len(claim_words & evidence_words) / len(claim_words)


# ----------------------------- Numeric checks -----------------------------
def extract_numbers(text):
    # Correctly matches integers, decimals (e.g. 37.31), and percentages.
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text or ""))


_METRIC_TERMS = (
    "rouge-1", "rouge-2", "rouge-l", "bleu", "accuracy",
    "precision", "recall", "f1", "f1-score", "score",
)


def has_numeric_metric_conflict(claim, evidence):
    claim_lower, evidence_lower = (claim or "").lower(), (evidence or "").lower()
    shared_metrics = [
        term for term in _METRIC_TERMS
        if term in claim_lower and term in evidence_lower
    ]
    claim_numbers, evidence_numbers = extract_numbers(claim), extract_numbers(evidence)
    return bool(
        shared_metrics and claim_numbers and evidence_numbers
        and claim_numbers.isdisjoint(evidence_numbers)
    )


# ----------------------------- Direct support -----------------------------
def direct_support(claim, evidence):
    """Conservative near-verbatim support check; paraphrases go through NLI."""
    if not normalize_text(claim) or not normalize_text(evidence):
        return False
    overlap = content_word_overlap(claim, evidence)
    if has_numeric_metric_conflict(claim, evidence):
        return False
    if overlap >= 0.80:
        return True
    claim_numbers, evidence_numbers = extract_numbers(claim), extract_numbers(evidence)
    return bool(
        claim_numbers and claim_numbers.issubset(evidence_numbers) and overlap >= 0.60
    )


# -------------------------- Relevant evidence search ----------------------
def find_relevant_sentences(claim, retrieved_chunks, top_k=8):
    """Return (sentence, reranker_score, lexical_score, retrieval_score)."""
    all_sentences, seen = [], set()
    for item in retrieved_chunks or []:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        chunk, retrieval_score = item[0], item[1]
        for sentence in split_sentences(str(chunk)):
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            all_sentences.append((
                sentence,
                content_word_overlap(claim, sentence),
                float(retrieval_score),
            ))
    if not all_sentences:
        return []

    pairs = [[claim, sentence] for sentence, _, _ in all_sentences]
    scores = reranker.predict(pairs)
    ranked = [
        (sentence, float(score), float(lexical), float(retrieval))
        for (sentence, lexical, retrieval), score in zip(all_sentences, scores)
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:max(1, top_k)]


# ----------------------------- Main analysis ------------------------------
def analyze_hallucination(answer, retrieved_chunks):
    """Classify claims as Supported, Contradicted, or Unverifiable."""
    if not answer or not retrieved_chunks or is_poor_answer(answer):
        return []

    claims = clean_claims(extract_claims(answer))
    if not claims:
        return []

    results = []
    for claim in claims:
        evidence_sentences = find_relevant_sentences(claim, retrieved_chunks, top_k=8)
        if not evidence_sentences:
            results.append({
                "claim": claim, "classification": "Unverifiable",
                "confidence": 0.0, "evidence": None,
            })
            continue

        # 1. Strong direct textual match.
        direct_evidence = next(
            (sentence for sentence, _, _, _ in evidence_sentences
             if direct_support(claim, sentence)),
            None,
        )
        if direct_evidence:
            results.append({
                "claim": claim, "classification": "Supported",
                "confidence": 1.0, "evidence": direct_evidence,
            })
            continue

        # 2. NLI on the best individual evidence sentences.
        supported, contradicted = [], []
        for sentence, reranker_score, lexical_score, retrieval_score in evidence_sentences:
            if lexical_score < 0.10 and reranker_score < 0.0:
                continue

            if has_numeric_metric_conflict(claim, sentence):
                contradicted.append((1.0, sentence, True))
                continue

            try:
                classification, confidence = classify_claim(claim, sentence)
                print("\nNLI DEBUG")
                print("Claim:", claim)
                print("Evidence:", sentence)
                print("NLI classification:", classification)
                print("NLI confidence:", confidence)
                print("Lexical overlap:", content_word_overlap(claim, sentence))
                classification = str(classification).strip().lower()
                confidence = float(confidence)
            except Exception as exc:
                print(f"Warning: NLI classification failed: {exc}")
                continue

            if classification == "supported":
                supported.append((confidence, sentence))
            elif classification == "contradicted":
                contradicted.append((confidence, sentence, False))

        if supported:
            confidence, evidence = max(supported, key=lambda item: item[0])
            results.append({
                "claim": claim, "classification": "Supported",
                "confidence": confidence, "evidence": evidence,
            })
            continue

        credible_contradictions = []
        for confidence, evidence, numeric_conflict in contradicted:
            overlap = content_word_overlap(claim, evidence)
            if numeric_conflict or (confidence >= 0.80 and overlap >= 0.20):
                credible_contradictions.append((confidence, evidence))

        if credible_contradictions:
            confidence, evidence = max(credible_contradictions, key=lambda item: item[0])
            results.append({
                "claim": claim, "classification": "Contradicted",
                "confidence": confidence, "evidence": evidence,
            })
        else:
            results.append({
                "claim": claim, "classification": "Unverifiable",
                "confidence": 0.0, "evidence": evidence_sentences[0][0],
            })

    return results


# ----------------------------- Metrics ------------------------------------
def calculate_metrics(results):
    total = len(results or [])
    if total == 0:
        return {"supported": 0.0, "contradicted": 0.0, "unverifiable": 0.0}

    supported = sum(r.get("classification") == "Supported" for r in results)
    contradicted = sum(r.get("classification") == "Contradicted" for r in results)
    unverifiable = sum(r.get("classification") == "Unverifiable" for r in results)
    return {
        "supported": supported / total * 100,
        "contradicted": contradicted / total * 100,
        "unverifiable": unverifiable / total * 100,
    }
