import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import extract_text_from_pdf, clean_text


def split_into_sections(text):
    """
    Detect major sections and subsections in the document.

    Examples detected:
        ABSTRACT
        I. INTRODUCTION
        II. LITERATURE REVIEW
        III. PROPOSED METHODOLOGY
        A. DATASET
        B. EXPERIMENT SETTING
        C. MODEL SETTING
        E. RESULTS AND DISCUSSION
        IV. EXPERIMENT AND RESULTS
        V. CONCLUSION
    """

    # Make sure ABSTRACT is treated as a heading
    text = re.sub(
        r"(?m)^ABSTRACT\s+",
        "ABSTRACT\n",
        text
    )

    # Detect:
    # 1. ABSTRACT
    # 2. Major sections such as III. PROPOSED METHODOLOGY
    # 3. Subsections such as E. RESULTS AND DISCUSSION
    #
    # The heading must:
    # - start at the beginning of a line
    # - contain uppercase letters
    # - have at least a few characters
    pattern = (
        r"(?m)^("
        r"ABSTRACT"
        r"|[A-Z]\.\s+[A-Z][A-Z0-9 ,:&()/-]{2,}"
        r"|[IVXLC]+\.\s+[A-Z][A-Z0-9 ,:&()/-]{2,}"
        r")\s*$"
    )

    matches = list(re.finditer(pattern, text))

    sections = []

    # If no headings are detected
    if not matches:
        return [("DOCUMENT", text)]

    # Text before the first heading
    if matches[0].start() > 0:

        preamble = text[:matches[0].start()].strip()

        if preamble:
            sections.append(
                ("DOCUMENT", preamble)
            )

    # Extract each section
    for i, match in enumerate(matches):

        section_title = match.group(1).strip()

        start = match.end()

        # End at the next heading
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        if section_text:
            sections.append(
                (
                    section_title,
                    section_text
                )
            )

    return sections


def split_text(text):
    """
    Cleanly split the document into overlapping chunks.

    Each chunk keeps its section name so that retrieval
    can identify where the information came from.
    """

    sections = split_into_sections(text)

    # RecursiveCharacterTextSplitter tries to split at
    # natural boundaries such as paragraphs and sentences.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )

    final_chunks = []

    for section_title, section_text in sections:

        chunks = splitter.split_text(
            section_text
        )

        for chunk in chunks:

            # Add section information to every chunk.
            enriched_chunk = (
                f"[SECTION: {section_title}]\n\n"
                f"{chunk}"
            )

            final_chunks.append(
                enriched_chunk
            )

    return final_chunks


if __name__ == "__main__":

    print("================================")
    print("CHUNKING TEST")
    print("================================")

    pdf_path = "sample.pdf"

    # Extract PDF text
    text = extract_text_from_pdf(
        pdf_path
    )

    print("\nRaw text extracted.")

    # Clean the extracted text
    cleaned_text = clean_text(
        text
    )

    print("Text cleaned.")

    # Detect sections
    sections = split_into_sections(
        cleaned_text
    )

    print(
        "\nTotal sections detected:",
        len(sections)
    )

    print("\nDetected sections:")

    for number, (title, section_text) in enumerate(
        sections,
        start=1
    ):
        print(
            f"{number}. {title}"
        )

    # Create chunks
    chunks = split_text(
        cleaned_text
    )

    print(
        "\nTotal chunks:",
        len(chunks)
    )

    # Display first few chunks
    print(
        "\n================================"
    )
    print("FIRST 3 CHUNKS")
    print("================================")

    for number, chunk in enumerate(
        chunks[:3],
        start=1
    ):
        print(
            f"\n--- CHUNK {number} ---"
        )
        print(chunk[:1000])