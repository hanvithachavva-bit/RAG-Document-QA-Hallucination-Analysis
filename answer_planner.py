
import re
from answer_requirements import parse_answer_requirements



def clean_question(query):
    return query.strip()



def detect_intent(question):
    """Identify the main purpose of the question."""

    q = question.lower().strip()

    # Summary questions, including questions about a document section.
    if any(term in q for term in (
        "what does the abstract say",
        "what does the conclusion say",
        "summarize",
        "summary",
        "main findings",
        "key findings",
        "main results",
        "key results",
        "overall findings",
        "overall results",
        "what did the study find",
        "what were the findings",
        "what does the paper conclude",
        "what is the main takeaway",
    )):
        return "summary"

    # Questions asking for a procedure or methodology.
    if any(term in q for term in (
        "step by step",
        "step-by-step",
        "steps involved",
        "what is the process",
        "how does",
        "how do",
        "how was",
        "how were",
        "how is",
        "how are",
        "how did",
        "how can",
    )):
        return "process"

    # Questions asking for reasons.
    if any(term in q for term in (
        "why ",
        "reason",
        "explain why",
    )):
        return "explanation"

    # Questions asking for specific items.
    if any(term in q for term in (
        "which models",
        "what models",
        "what are the models",
        "name the models",
        "which methods",
        "what methods",
        "which techniques",
        "what techniques",
        "what metrics",
        "which metrics",
        "what types",
        "which types",
        "list the",
        "list out",
        "name the",
    )):
        return "list"

    # Definitions, not questions about what a section says.
    if any(term in q for term in (
        "define ",
        "definition of",
        "meaning of",
        "what is ",
        "what are ",
    )):
        return "definition"

    if "what does " in q:
        return "general"

    if any(term in q for term in (
        "explain ",
        "describe ",
        "elaborate ",
    )):
        return "explanation"

    return "general"


def detect_subject(question):
    """Identify relevant subject categories without assuming answers."""

    q = question.lower()

    subjects = {
        "models": ["models", "model"],
        "metrics": ["evaluation metrics", "metrics", "metric"],
        "findings": ["findings", "results", "outcomes"],
        "classification": ["classification"],
        "summarization": ["summarization", "summary"],
        "dataset": ["dataset", "datasets"],
        "performance": [
            "performance", "accuracy", "precision",
            "recall", "f1-score", "f1",
        ],
        "methodology": [
            "methodology", "approach", "framework", "method",
        ],
        "experiment": [
            "experiment", "experimental setup", "experimental setting",
        ],
    }

    detected = []

    for subject, keywords in subjects.items():
        if any(keyword in q for keyword in keywords):
            detected.append(subject)

    return detected or ["general"]


def detect_focus(intent, subjects):
    """Describe the evidence that should be prioritized."""

    if intent == "summary":
        return "main findings and important results"

    if intent == "process":
        return "process stages and methodology"

    if intent == "explanation":
        return "explanation and supporting evidence"

    if intent == "definition":
        return "definition and key characteristics"

    if intent == "list":
        if "models" in subjects:
            return "requested models"
        if "metrics" in subjects:
            return "requested evaluation metrics"
        if "dataset" in subjects:
            return "requested dataset information"
        return "requested items"

    return "information directly relevant to the question"


def create_answer_plan(query):
    """
    Combine the cleaned question, intent, subject, and answer
    requirements into a single structured plan.
    """

    requirements = parse_answer_requirements(query)
    question = clean_question(query)

    if not question:
        question = query.strip()

    intent = detect_intent(question)
    subjects = detect_subject(question)
    focus = detect_focus(intent, subjects)

    return {
        **requirements,
        "original_question": query,
        "question": question,
        "clean_question": question,
        "intent": intent,
        "subjects": subjects,
        "focus": focus,
    }


if __name__ == "__main__":
    tests = [
        "Which models were selected for the sentence classification task?",
        "Which models were selected for the sentence classification task? Explain them in simple terms in 100-150 words.",
        "What models were selected? Give me the answer in 5-10 points.",
        "What were the main findings? Summarize briefly.",
        "How was the classification task performed? Explain step by step.",
        "Why was LLaMA2 used for sentence classification? Give a short answer.",
    ]

    for query in tests:
        print("\n" + "=" * 60)
        print("INPUT:", query)
        print("PLAN:")
        for key, value in create_answer_plan(query).items():
            print(f"{key}: {value}")
