# RAG-Based Document Q&A System with Hallucination Analysis

A Retrieval-Augmented Generation (RAG) application that answers questions about uploaded PDF documents and analyzes generated answers to identify claims that are supported, contradicted, or unverifiable based on retrieved evidence.

## Overview

Large Language Models (LLMs) can generate fluent responses that are not always supported by source documents. This project combines document retrieval with answer generation and claim-level hallucination analysis to make answers more evidence-grounded and their limitations more visible.

Users can upload a PDF, ask questions about its contents, view the generated answer, and review the claim-analysis results through a Streamlit interface.

## Key Features

- **PDF Processing:** Extracts text from uploaded PDF documents.
- **Text Chunking:** Splits document text into overlapping chunks of 1,000 characters with a 300-character overlap.
- **Embedding Generation:** Uses `BAAI/bge-small-en-v1.5` to convert text chunks into 384-dimensional vectors.
- **Vector Search:** Uses FAISS for similarity-based retrieval.
- **Hybrid Retrieval:** Combines semantic similarity with keyword matching.
- **Reranking:** Uses a cross-encoder to improve the relevance of retrieved passages.
- **Answer Generation:** Uses a Qwen-based language model to generate context-aware answers.
- **Hallucination Analysis:** Extracts individual claims and classifies them as Supported, Contradicted, or Unverifiable using evidence checks and Natural Language Inference (NLI).
- **Interactive Interface:** Uses Streamlit for document upload, question entry, answer display, and claim-analysis results.
- **Evaluation Support:** Includes modules for testing and comparing RAG and non-RAG answer generation.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Text Extraction | PyMuPDF |
| Text Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector Search | FAISS |
| Reranking | Cross-encoder |
| Language Model | Qwen-based LLM |
| Claim Analysis | Evidence checks and NLI |
| Deep Learning Framework | PyTorch |
| Model Loading | Hugging Face Transformers |

## System Workflow

1. Upload a PDF document through the Streamlit interface.
2. Extract text from the PDF.
3. Split the extracted text into overlapping chunks.
4. Generate embeddings for the chunks using BGE.
5. Index the embeddings using FAISS.
6. Retrieve relevant passages using semantic search and keyword matching.
7. Rerank candidate passages using a cross-encoder.
8. Pass the question and selected context to the Qwen-based language model.
9. Generate an answer using the retrieved evidence.
10. Extract factual claims from the answer.
11. Analyze the claims against available evidence and classify them as Supported, Contradicted, or Unverifiable.
12. Display the answer and claim-analysis results in the interface.

## Project Structure

```text
RAG-Document-QA/
├── app.py
├── document_loader.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retrieval.py
├── reranker.py
├── llm.py
├── rag_pipeline.py
├── claim_extractor.py
├── nli_classifier.py
├── hallucination_analyzer.py
├── answer_evaluator.py
├── answer_planner.py
├── answer_requirements.py
├── baseline.py
├── compare.py
├── evaluation.py
├── evaluation_questions.py
├── document_manager.py
├── document_query.py
├── document_store.py
├── stored_document.py
├── stored_rag_pipeline.py
├── stored_retrieval.py
├── test_hallucination.py
├── test_question.py
├── debug_retrieval.py
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.13 or a compatible Python version supported by the project's dependencies
- Git
- Internet access to download required model files, if they are not already available locally

### 1. Clone the repository

```bash
git clone https://github.com/hanvithachavva-bit/RAG-Document-QA-Hallucination-Analysis.git
cd RAG-Document-QA-Hallucination-Analysis
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the libraries required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

If it is not available yet, create one containing the project's actual dependencies before running the application. Avoid installing arbitrary package versions without checking compatibility.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Hallucination Analysis

The system analyzes individual claims in generated answers against retrieved evidence.

- **Supported:** The available evidence supports the claim.
- **Contradicted:** The available evidence conflicts with the claim.
- **Unverifiable:** The available evidence is insufficient to confirm or contradict the claim.

This analysis helps users understand how generated statements relate to the source material. It does not guarantee that every answer is correct or eliminate all hallucinations.

## Evaluation

The project includes evaluation-related modules and support for comparing RAG-based responses with a non-RAG baseline. The goal is to assess whether retrieving document evidence helps reduce unsupported claims.

Evaluation results should be reported only after running the tests and analyzing the actual outputs.

## Limitations

- Answer quality depends on the quality of the uploaded document and retrieved passages.
- Relevant evidence may be missed during retrieval.
- The language model may still generate unsupported statements.
- Claim extraction and NLI classification can make mistakes.
- Model downloads and inference can require significant memory and processing time, depending on configuration.

## Future Enhancements

- Improve retrieval and reranking quality.
- Expand evaluation datasets and metrics.
- Improve claim-level evidence presentation.
- Add document citation and source-location display.
- Deploy the application for public access.
- Improve error handling and document validation.

## Project Ownership

Independently implemented the RAG pipeline, document retrieval, answer generation, and claim-level hallucination analysis as part of an academic minor degree project.

## Disclaimer

This project is intended for academic and research purposes. Hallucination-analysis labels indicate relationships between claims and available evidence; they should not be treated as a guarantee of factual correctness.
