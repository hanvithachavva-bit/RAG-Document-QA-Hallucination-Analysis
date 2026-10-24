
from hallucination_analyzer import analyze_hallucination, calculate_metrics

evidence = [
    (
        "[SECTION: RESULTS] LLaMA2-13B achieved a ROUGE-1 score of 37.31.",
        1.0
    ),
    (
        "[SECTION: RESULTS] The experiment used 60 manually annotated resumes.",
        1.0
    )
]

test_cases = [
    "LLaMA2-13B achieved a ROUGE-1 score of 37.31.",
    "LLaMA2-13B achieved a ROUGE-1 score of 50.00.",
    "The experiment was conducted in 2024."
]

for claim in test_cases:
    print("\n" + "=" * 60)
    print("CLAIM:", claim)

    results = analyze_hallucination(claim, evidence)
    print("RESULT:", results)
    print("METRICS:", calculate_metrics(results))
