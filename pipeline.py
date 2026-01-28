from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import re
import unicodedata

import numpy as np
import fitz  # PyMuPDF
import faiss

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class Chunk:
    chunk_id: int
    page: int  # 1-indexed
    text: str


@dataclass
class Bullet:
    text: str
    pages: List[int]
    evidence_quotes: List[str]
    support_score: float
    support_label: str


def extract_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Tuple[int, str]] = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append((i + 1, text))
    return pages


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ord(ch) >= 32))
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return s.strip()


def clean_generated_text(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"\.{3,}", "...", s)
    s = re.sub(r"([!?]){2,}", r"\1", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_pages(pages: List[Tuple[int, str]], max_chunk_chars: int = 900) -> List[Chunk]:
    chunks: List[Chunk] = []
    cid = 0

    for page_no, raw in pages:
        text = clean_text(raw)
        if not text:
            continue

        sentences = split_into_sentences(text)
        if not sentences:
            continue

        current = ""
        for sent in sentences:
            if len(sent) > max_chunk_chars:
                words = sent.split()
                buf = ""
                for w in words:
                    if len(buf) + len(w) + 1 <= max_chunk_chars:
                        buf = (buf + " " + w).strip()
                    else:
                        if buf:
                            chunks.append(Chunk(cid, page_no, buf))
                            cid += 1
                        buf = w
                if buf:
                    chunks.append(Chunk(cid, page_no, buf))
                    cid += 1
                current = ""
                continue

            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= max_chunk_chars:
                current = current + " " + sent
            else:
                chunks.append(Chunk(cid, page_no, current.strip()))
                cid += 1
                current = sent

        if current.strip():
            chunks.append(Chunk(cid, page_no, current.strip()))
            cid += 1

    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    emb = embeddings.astype("float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def retrieve_ids(query_vec: np.ndarray, index: faiss.IndexFlatIP, top_k: int) -> List[int]:
    q = query_vec.astype("float32")
    faiss.normalize_L2(q)
    _, ids = index.search(q, top_k)
    return [int(i) for i in ids[0] if i != -1]


def make_evidence_quotes(chunks: List[Chunk], max_quote_chars: int = 280) -> List[str]:
    quotes: List[str] = []
    for c in chunks:
        t = clean_text(c.text).replace("\n", " ")
        if len(t) > max_quote_chars:
            t = t[:max_quote_chars].rsplit(" ", 1)[0] + "…"
        quotes.append(t)
    return quotes


def support_score(bullet_text: str, evidence_chunks: List[Chunk], embedder: SentenceTransformer) -> float:
    texts = [bullet_text] + [c.text for c in evidence_chunks]
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    b = vecs[0]
    ev = vecs[1:]
    if ev.shape[0] == 0:
        return 0.0
    sims = (ev @ b).tolist()
    return float(max(sims))


def label_support(score: float) -> str:
    if score >= 0.55:
        return "Supported"
    if score >= 0.42:
        return "Weakly supported"
    return "Potentially unsupported"


def dedupe_chunks(chunks: List[Chunk]) -> List[Chunk]:
    seen = set()
    out: List[Chunk] = []
    for c in chunks:
        key = clean_text(c.text)
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def diversify_by_page(candidates: List[Chunk], desired_k: int) -> List[Chunk]:
    """
    Prefer evidence across different pages when possible.
    Greedy selection:
      - pick first chunk
      - then prefer chunks from new pages
      - then fill remaining
    """
    if len(candidates) <= desired_k:
        return candidates

    selected: List[Chunk] = []
    used_pages = set()

    # First pass: new pages
    for c in candidates:
        if c.page not in used_pages:
            selected.append(c)
            used_pages.add(c.page)
        if len(selected) >= desired_k:
            return selected

    # Second pass: fill
    for c in candidates:
        if c not in selected:
            selected.append(c)
        if len(selected) >= desired_k:
            break

    return selected


class SummarisationSystem:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        summariser_model: str = "facebook/bart-large-cnn",
        device: str | None = None,
    ):
        self.embedder = SentenceTransformer(embed_model)

        # Cross-encoder reranker (CPU is fine for reranking ~20 pairs)
        self.reranker = CrossEncoder(reranker_model)

        self.sum_tokenizer = AutoTokenizer.from_pretrained(summariser_model)
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(summariser_model)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sum_model.to(self.device)
        self.sum_model.eval()

    def _safe_summarise(
        self,
        text: str,
        max_input_tokens: int = 900,
        max_new_tokens: int = 90,
        min_new_tokens: int = 35,
    ) -> str:
        text = clean_text(text)
        if not text:
            return ""

        inputs = self.sum_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self.sum_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=6,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
            )

        out = self.sum_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return clean_generated_text(out)

    def index_pdf(self, pdf_bytes: bytes, chunk_size: int = 900, overlap: int = 150) -> Dict[str, Any]:
        pages = extract_pdf_pages(pdf_bytes)
        chunks = chunk_pages(pages, max_chunk_chars=chunk_size)
        if not chunks:
            raise ValueError("No extractable text found in the PDF (may be scanned images).")

        embeddings = self.embedder.encode([c.text for c in chunks], convert_to_numpy=True)
        index = build_faiss_index(np.asarray(embeddings))
        return {"pages": pages, "chunks": chunks, "embeddings": embeddings, "index": index}

    def baseline_summary(self, full_text: str) -> str:
        return self._safe_summarise(full_text, max_input_tokens=900, max_new_tokens=180, min_new_tokens=60)

    def _retrieve_and_rerank(
        self,
        query: str,
        index_bundle: Dict[str, Any],
        retrieve_k: int,
        final_k: int,
    ) -> List[Chunk]:
        chunks: List[Chunk] = index_bundle["chunks"]
        index = index_bundle["index"]

        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        ids = retrieve_ids(q_vec, index, top_k=retrieve_k)
        candidates = [chunks[i] for i in ids]
        candidates = dedupe_chunks(candidates)

        if not candidates:
            return []

        # Rerank with cross-encoder
        pairs = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        ranked = diversify_by_page(ranked, desired_k=final_k)

        return ranked[:final_k]

    def rag_bullets(
        self,
        query_list: List[str],
        index_bundle: Dict[str, Any],
        top_k: int = 5,
        bullet_max_new_tokens: int = 90,
        bullet_min_new_tokens: int = 35,
        max_evidence_input_tokens: int = 900,
        retrieve_k: int = 20,
    ) -> List[Bullet]:
        bullets: List[Bullet] = []

        for q in query_list:
            ev_chunks = self._retrieve_and_rerank(
                query=q,
                index_bundle=index_bundle,
                retrieve_k=retrieve_k,
                final_k=top_k,
            )

            if not ev_chunks:
                bullets.append(
                    Bullet(
                        text=f"(No evidence retrieved for: {q})",
                        pages=[],
                        evidence_quotes=[],
                        support_score=0.0,
                        support_label="Potentially unsupported",
                    )
                )
                continue

            evidence_text = "\n\n".join([c.text for c in ev_chunks])
            summary = self._safe_summarise(
                evidence_text,
                max_input_tokens=max_evidence_input_tokens,
                max_new_tokens=bullet_max_new_tokens,
                min_new_tokens=bullet_min_new_tokens,
            )

            pages = sorted({c.page for c in ev_chunks})
            quotes = make_evidence_quotes(ev_chunks)

            score = support_score(summary, ev_chunks, self.embedder) if summary else 0.0
            label = label_support(score) if summary else "Potentially unsupported"

            bullets.append(
                Bullet(
                    text=summary if summary else "(Empty summary output)",
                    pages=pages,
                    evidence_quotes=quotes,
                    support_score=score,
                    support_label=label,
                )
            )

        return bullets