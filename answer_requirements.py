
import re


def parse_answer_requirements(query):
    """
    Extract the user's answer constraints into a structured dictionary.

    This function identifies constraints; answer_planner.py will
    combine them with the actual question and its intent.
    """

    text = query.strip()
    q = text.lower()

    requirements = {
        "marks": None,
        "min_words": None,
        "max_words": None,
        "min_points": None,
        "max_points": None,
        "detail_level": "normal",
        "format": "paragraph",
        "style": "normal",
        "constraints": [],
    }

    # --------------------------------------------------
    # MARKS
    # --------------------------------------------------
    match = re.search(
        r"\b(\d+)\s*[-–]?\s*marks?\b", q
    )

    if match:
        requirements["marks"] = int(match.group(1))

    # --------------------------------------------------
    # WORD COUNT
    # --------------------------------------------------
    match = re.search(
        r"\b(\d+)\s*[-–]\s*(\d+)\s*words?\b", q
    )

    if match:
        low, high = int(match.group(1)), int(match.group(2))
        requirements["min_words"] = min(low, high)
        requirements["max_words"] = max(low, high)
    else:
        match = re.search(r"\b(\d+)\s*words?\b", q)
        if match:
            requirements["max_words"] = int(match.group(1))

    # --------------------------------------------------
    # NUMBER OF POINTS
    # --------------------------------------------------
    point_pattern = (
        r"\b(\d+)\s*[-–]\s*(\d+)\s+"
        r"(?:points?|bullet points?|items?)\b"
    )
    match = re.search(point_pattern, q)

    if match:
        low, high = int(match.group(1)), int(match.group(2))
        requirements["min_points"] = min(low, high)
        requirements["max_points"] = max(low, high)
    else:
        match = re.search(
            r"\b(?:in|as|give me|give the answer in)\s+"
            r"(\d+)\s+(?:points?|bullet points?|items?)\b",
            q,
        )
        if match:
            requirements["min_points"] = int(match.group(1))
            requirements["max_points"] = int(match.group(1))

    # --------------------------------------------------
    # DETAIL LEVEL
    # --------------------------------------------------
    if any(phrase in q for phrase in (
        "in detail",
        "in depth",
        "in-depth",
        "thoroughly",
        "comprehensive explanation",
        "elaborate on",
        "explain completely",
        "detailed explanation",
        "detailed answer",
    )):
        requirements["detail_level"] = "detailed"

    elif any(phrase in q for phrase in (
        "briefly",
        "in brief",
        "short answer",
        "brief answer",
        "summarize briefly",
        "keep it short",
        "be concise",
    )):
        requirements["detail_level"] = "short"

    elif any(phrase in q for phrase in (
        "moderate answer",
        "medium-length answer",
        "explain moderately",
    )):
        requirements["detail_level"] = "medium"

    # Marks influence detail only when no explicit detail instruction
    explicit_detail = requirements["detail_level"] != "normal"

    if requirements["marks"] is not None and not explicit_detail:
        marks = requirements["marks"]

        if marks <= 2:
            requirements["detail_level"] = "short"
        elif marks <= 5:
            requirements["detail_level"] = "medium"
        else:
            requirements["detail_level"] = "detailed"

    if (
        requirements["min_words"] is not None
        or requirements["max_words"] is not None
    ):
        requirements["detail_level"] = "detailed"

    # --------------------------------------------------
    # OUTPUT FORMAT
    # --------------------------------------------------
    if any(phrase in q for phrase in (
        "step by step",
        "step-by-step",
        "stepwise",
        "step wise",
        "in steps",
        "as steps",
        "numbered list",
        "numbered steps",
    )):
        requirements["format"] = "steps"

    elif any(phrase in q for phrase in (
        "in points",
        "point-wise",
        "point wise",
        "pointwise",
        "bullet points",
        "in bullets",
        "as bullets",
        "as a list",
        "in a list",
        "list the points",
    )):
        requirements["format"] = "bullets"

    # A requested point count implies a point-based format.
    if requirements["max_points"] is not None:
        requirements["format"] = "bullets"

    # --------------------------------------------------
    # LANGUAGE / STYLE
    # --------------------------------------------------
    if any(phrase in q for phrase in (
        "in simple terms",
        "in simple words",
        "simply explain",
        "explain simply",
        "easy to understand",
        "for a beginner",
    )):
        requirements["style"] = "simple"

    elif any(phrase in q for phrase in (
        "formally",
        "in formal language",
        "academic language",
    )):
        requirements["style"] = "formal"

    # --------------------------------------------------
    # EVIDENCE CONSTRAINTS
    # --------------------------------------------------
    if any(phrase in q for phrase in (
        "only use information from the document",
        "use only the document",
        "based only on the document",
        "only use the provided context",
        "do not use outside knowledge",
    )):
        requirements["constraints"].append("document_only")

    return requirements
