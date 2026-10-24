import re


def clean_generated_artifacts(text):
    """
    Remove common artifacts produced by the language model.
    """

    # Remove special generation markers
    artifacts = [
        "[END_OF_TEXT]",
        "[END OF TEXT]",
        "<|endoftext|>",
        "<|end_of_text|>",
        "</s>"
    ]

    for artifact in artifacts:
        text = text.replace(
            artifact,
            ""
        )

    # Remove accidental trailing incomplete phrases
    text = re.sub(
        r"\bTo determine if the given text contains relevant information about.*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def extract_claims(answer):
    """
    Extract factual claims from a generated answer.

    Each sufficiently long sentence is treated as one
    candidate claim.
    """

    if not answer:
        return []

    # --------------------------------------------------------
    # Clean model-generated artifacts
    # --------------------------------------------------------

    answer = clean_generated_artifacts(
        answer
    )

    if not answer:
        return []

    # --------------------------------------------------------
    # Split answer into sentences
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    claims = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Remove leading bullets
        sentence = re.sub(
            r"^[\-\*\•]\s*",
            "",
            sentence
        )

        # Remove citation numbers
        sentence = re.sub(
            r"\s*\[\d+(?:,\s*\d+)*\]",
            "",
            sentence
        )

        sentence = sentence.strip()

        # Ignore very short fragments
        if len(sentence.split()) < 5:
            continue

        # Ignore incomplete fragments
        incomplete_phrases = [
            "these results highlight",
            "these results demonstrate",
            "the results show"
        ]

        if sentence.lower().rstrip(".") in incomplete_phrases:
            continue

        claims.append(
            sentence
        )

    return claims


if __name__ == "__main__":

    test_answer = """
    The model achieved strong performance.
    The results showed significant improvement over the baseline.
    """

    print("Extracted claims:")

    claims = extract_claims(
        test_answer
    )

    for number, claim in enumerate(
        claims,
        start=1
    ):

        print(
            number,
            claim
        )