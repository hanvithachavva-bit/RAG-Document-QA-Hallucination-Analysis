import pymupdf
import re


def extract_text_from_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text() + "\n"

    document.close()

    return text


def clean_text(text):

    # ========================================================
    # REMOVE IEEE PAGE HEADERS
    # ========================================================

    text = re.sub(
        r"D\. Peng: Deep Learning-Driven Labor Education and Skill Assessment: A Big Data Approach",
        "",
        text
    )

    # ========================================================
    # REMOVE PAGE NUMBERS
    # ========================================================

    text = re.sub(
        r"\n\d{6}\n",
        "\n",
        text
    )

    # ========================================================
    # REMOVE VOLUME/YEAR FOOTER
    # ========================================================

    text = re.sub(
        r"VOLUME 13, 2025",
        "",
        text
    )

    # ========================================================
    # REMOVE COPYRIGHT/LICENSE TEXT
    # ========================================================

    text = re.sub(
        r"2025 The Authors\. This work is licensed under a Creative Commons Attribution 4\.0 License\.",
        "",
        text
    )

    # ========================================================
    # REMOVE PUBLICATION METADATA
    # ========================================================

    text = re.sub(
        r"Received .*?Digital Object Identifier .*?\n",
        "",
        text,
        flags=re.DOTALL
    )

    # ========================================================
    # REMOVE AUTHOR AFFILIATION AND EMAIL
    # ========================================================

    text = re.sub(
        r"DAN PENG.*?e-mail:.*?\n",
        "",
        text,
        flags=re.DOTALL
    )

    # ========================================================
    # REMOVE EDITOR NOTE
    # ========================================================

    text = re.sub(
        r"The associate editor coordinating.*?publication\s*\.",
        "",
        text,
        flags=re.DOTALL
    )

    # ========================================================
    # REMOVE CREATIVE COMMONS NOTICE
    # ========================================================

    text = re.sub(
        r"For more information, see https://creativecommons\.org/licenses/by/4\.0/",
        "",
        text
    )

    # ========================================================
    # FIX HYPHENATED WORDS BROKEN ACROSS LINES
    # Example:
    # classifi-
    # cation
    # becomes:
    # classification
    # ========================================================

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # ========================================================
    # FIX MISSING SPACES AFTER PDF EXTRACTION
    #
    # Examples:
    # predefinedclasses -> predefined classes
    # sentenceclassification -> sentence classification
    # acrossdifferent -> across different
    # ========================================================

    common_word_fixes = {
        "predefinedclasses": "predefined classes",
        "predefinedcategories": "predefined categories",
        "sentenceclassification": "sentence classification",
        "sentencesclassification": "sentences classification",
        "resumeclassification": "resume classification",
        "resumegrading": "resume grading",
        "wassignificantly": "was significantly",
        "acrossdifferent": "across different",
        "informationextraction": "information extraction",
        "candidateevaluation": "candidate evaluation",
        "workexperience": "work experience",
        "personalinformation": "personal information",
        "stateoftheart": "state-of-the-art",
        "fine-tuningprocess": "fine-tuning process",
        "downstreamtasks": "downstream tasks",
        "inputsequence": "input sequence",
        "querysentence": "query sentence",

    # Additional PDF extraction corrections
        "resumescreening": "resume screening",
        "componentof": "component of",
        "fordecision-making": "for decision-making",
        "improvesboth": "improves both",
        "screeningprocess": "screening process",
        "isperformed": "is performed",
        "theselection": "the selection",
        "resumesare": "resumes are",
        "organizationswhere": "organizations where",
        "mustbe": "must be",
        "LLMsfor": "LLMs for",
        "modelachieves": "model achieves",
        "significantlyoutperforming": "significantly outperforming",
    }

    for incorrect, correct in common_word_fixes.items():
        text = re.sub(
            re.escape(incorrect),
            correct,
            text,
            flags=re.IGNORECASE
        )

    # ========================================================
    # NORMALIZE SPACES
    # ========================================================

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # ========================================================
    # NORMALIZE BLANK LINES
    # ========================================================

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# TEST THE FILE DIRECTLY
# ============================================================

if __name__ == "__main__":

    pdf_path = "sample.pdf"

    text = extract_text_from_pdf(pdf_path)

    cleaned_text = clean_text(text)

    print(cleaned_text[:5000])