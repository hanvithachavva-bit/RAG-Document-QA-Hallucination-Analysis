import re



def evaluate_answer_quality(answer):

    answer = answer.strip()

    if not answer:
        return "No Answer"

    answer_lower = answer.lower()

    # Explicit no-answer responses
    fallback_phrases = [
        "the document does not provide enough information",
        "the selected document does not contain enough relevant information",
        "not enough information to answer",
        "the context does not contain the answer",
        "i do not have enough information",
        "cannot answer",
        "unable to answer"
    ]

    for phrase in fallback_phrases:
        if phrase in answer_lower:
            return "No Answer"

    # Extract words
    words = re.findall(
        r"[a-zA-Z0-9]+(?:[-][a-zA-Z0-9]+)*",
        answer_lower
    )

    if len(words) < 2:
        return "Incomplete"

    if len(words) < 5:
        return "Incomplete"

    # Detect excessive single-letter output
    single_letter_count = sum(
        1 for word in words if len(word) == 1
    )

    if (
        len(words) >= 6
        and single_letter_count / len(words) >= 0.4
    ):
        return "Poor"

    # Detect excessive repetition
    unique_word_ratio = len(set(words)) / len(words)

    if len(words) >= 8 and unique_word_ratio < 0.55:
        return "Poor"

    # Normalize punctuation and whitespace
    normalized = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        answer_lower
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Detect repeated consecutive phrases
    for phrase_size in range(3, 7):
        tokens = normalized.split()

        for i in range(len(tokens) - 2 * phrase_size + 1):
            first = tokens[i:i + phrase_size]
            second = tokens[i + phrase_size:i + 2 * phrase_size]

            if first == second:
                return "Poor"

    # Detect obviously incoherent or irrelevant output
    nonsense_phrases = [
        "the syringes were able to be inserted into the syringes",
        "it is a good idea i think it is a good idea",
    ]

    if any(phrase in normalized for phrase in nonsense_phrases):
        return "Poor"

    return "Good"


if __name__ == "__main__":

    test_answers = [

        "LLaMA2-7B, Falcon, FLAN-T5, and GPT-NeoX.",

        "The LLaMA2 model.",

        "The document does not provide enough information to answer this question.",

        "a b c d e f g h i j k m n",

        "The proposed system integrates information extraction and summarization tasks.",

        "it is a good idea. i think it is a good idea. i think it is a good idea."
    ]

    print("\nANSWER QUALITY TEST\n")

    for answer in test_answers:

        quality = evaluate_answer_quality(
            answer
        )

        print("Answer:")
        print(answer)

        print("Quality:", quality)

        print("----------------------------")