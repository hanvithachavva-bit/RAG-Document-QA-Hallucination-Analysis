evaluation_questions = [
    {
        "question": "What does the abstract say about the proposed system?",
        "section": "ABSTRACT"
    },
    {
        "question": "What is the main focus of the research?",
        "section": "I. INTRODUCTION"
    },
    {
        "question": "What challenges are discussed in the literature review?",
        "section": "II. LITERATURE REVIEW"
    },
    {
        "question": "What methodology does the proposed system use?",
        "section": "III. PROPOSED METHODOLOGY"
    },
    {
        "question": "What are the main results reported by the research?",
        "section": "EXPERIMENT AND RESULTS"
    }
]


if __name__ == "__main__":

    print("Evaluation Questions:\n")

    for i, item in enumerate(
        evaluation_questions,
        start=1
    ):

        print(
            f"{i}. {item['question']}"
        )

        print(
            f"   Section: {item['section']}"
        )