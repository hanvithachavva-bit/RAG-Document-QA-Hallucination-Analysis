
from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME
)


def classify_claim(claim, context):
    """
    Compare a claim against its evidence.

    context = premise/evidence from the document
    claim   = hypothesis being checked
    """

    result = classifier(
        {
            "text": context,
            "text_pair": claim
        },
        truncation=True
    )

    # Handle both a single-result dictionary and a list of results.
    if isinstance(result, list):
        result = result[0]

    label = result["label"].strip().lower()
    confidence = float(result["score"])

    # Handle models that return generic labels such as LABEL_0.
    if label.startswith("label_"):
        label_id = label.split("_")[-1]

        # Verify the mapping against the model's own configuration.
        id2label = classifier.model.config.id2label
        configured_label = id2label.get(
            int(label_id),
            label
        ).strip().lower()

        label = configured_label

    if "entailment" in label:
        classification = "Supported"

    elif "contradiction" in label:
        classification = "Contradicted"

    elif "neutral" in label:
        classification = "Unverifiable"

    else:
        classification = "Unverifiable"

    return classification, confidence


if __name__ == "__main__":
    context = (
        "The proposed system uses LLM models for automating "
        "resume screening and grading. The system integrates "
        "information extraction and summarization tasks."
    )

    claims = [
        "The proposed system uses LLMs for automated resume screening and grading.",
        "The system integrates information extraction and summarization tasks.",
        "The system uses blockchain for resume verification."
    ]

    print("\nNLI Results:\n")

    for claim in claims:
        classification, confidence = classify_claim(
            claim,
            context
        )

        print("Claim:", claim)
        print("Classification:", classification)
        print("Confidence:", round(confidence, 4))
        print()
