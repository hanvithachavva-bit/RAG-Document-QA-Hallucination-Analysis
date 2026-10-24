from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from claim_extractor import extract_claims
from nli_classifier import classify_claim


MODEL_NAME = "google/flan-t5-base"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_non_rag_answer(query):

    prompt = f"""
Answer the question as accurately as possible.

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=20
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


def analyze_baseline(answer, context):

    claims = extract_claims(answer)

    results = []

    for claim in claims:

        classification, confidence = classify_claim(
            claim,
            context
        )

        results.append({
            "claim": claim,
            "classification": classification,
            "confidence": confidence
        })

    return results


if __name__ == "__main__":

    query = input(
        "Enter your question: "
    )

    answer = generate_non_rag_answer(
        query
    )

    print("\n================================")
    print("NON-RAG ANSWER")
    print("================================")

    print(answer)

    print("\n================================")
    print("NON-RAG CLAIM ANALYSIS")
    print("================================")

    # For the baseline, we use the document as evidence
    # only for evaluation. The LLM itself did NOT receive it.
    context = """
    The proposed system integrates information extraction
    and summarization tasks, leveraging LLMs for decision-making
    throughout the hiring process.
    """

    results = analyze_baseline(
        answer,
        context
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nClaim {i}:")
        print(result["claim"])

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