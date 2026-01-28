# Traceable PDF Summariser

A Streamlit prototype that generates traceable study summaries from PDFs using retrieval-augmented generation (RAG).

## Features
- Upload PDF and extract text per page
- Chunk + embed content and build a FAISS index
- Generate topic-based RAG bullet summaries (BART)
- Page-level citations + evidence quotes
- Support score with donut chart
- Baseline (non-grounded) summary for comparison

## Setup
```bash
python -m pip install -r requirements.txt

## Run
python -m streamlit run app.py
