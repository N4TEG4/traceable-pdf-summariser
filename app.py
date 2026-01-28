import streamlit as st
import plotly.graph_objects as go

from pipeline import SummarisationSystem, clean_text


st.set_page_config(page_title="Traceable PDF Summariser (RAG + Support Check)", layout="wide")

st.title("Traceable PDF Summariser")
st.caption("RAG summaries with page citations + evidence quotes + support scoring (no PDF highlighting).")


@st.cache_resource
def load_system():
    return SummarisationSystem()


def support_donut(score: float, label: str):
    # Clamp score for display
    s = max(0.0, min(1.0, float(score)))

    fig = go.Figure(
        go.Pie(
            values=[s, 1 - s],
            labels=["Supported", "Remaining"],
            hole=0.65,
            textinfo="none",
            sort=False,
            direction="clockwise",
        )
    )
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=180)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Support score: {s:.2f} • {label}")


sys = load_system()

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

with st.sidebar:
    st.header("Settings")

    # Chunking: now sentence-ish chunking uses chunk_size as max chars per chunk.
    chunk_size = st.slider("Chunk size (max chars per chunk)", 400, 1400, 900, 50)

    # Overlap kept for API compatibility; not used by sentence chunker.
    overlap = st.slider("Overlap (unused in current chunker)", 0, 400, 150, 25)

    top_k = st.slider("Evidence chunks per bullet (top-k)", 2, 8, 5, 1)

    st.divider()
    st.subheader("Bullet topics (queries)")
    st.write("One per line. These guide retrieval for each bullet.")
    default_queries = [
        "Key concepts and definitions",
        "Main arguments or important points",
        "Methods / process described",
        "Results / conclusions",
        "Limitations or cautions",
    ]
    queries = st.text_area("Queries", value="\n".join(default_queries), height=160)
    query_list = [q.strip() for q in queries.splitlines() if q.strip()]

if uploaded is None:
    st.stop()

pdf_bytes = uploaded.read()

try:
    with st.spinner("Indexing PDF (extracting text, chunking, embedding, building search index)…"):
        bundle = sys.index_pdf(pdf_bytes, chunk_size=chunk_size, overlap=overlap)
except Exception as e:
    st.error(f"Failed to process PDF: {e}")
    st.stop()

# Full text for baseline + preview
full_text = "\n\n".join([t for _, t in bundle["pages"]])
full_text = clean_text(full_text)

tabs = st.tabs(["Traceable RAG bullets", "Baseline summary", "Extraction preview"])

with tabs[0]:
    st.subheader("RAG bullets (traceable)")

    with st.spinner("Generating grounded bullets…"):
        bullets = sys.rag_bullets(query_list, bundle, top_k=top_k)

    for i, b in enumerate(bullets, start=1):
        with st.expander(f"Bullet {i}: {b.support_label}", expanded=True):
            left, right = st.columns([3, 1])

            with left:
                st.markdown("**Summary**")
                st.write(b.text)

                if b.pages:
                    st.markdown("**Cited pages**")
                    st.write(", ".join([f"p.{p}" for p in b.pages]))
                else:
                    st.markdown("**Cited pages**")
                    st.write("None")

                st.markdown("**Evidence quotes**")
                if b.evidence_quotes:
                    for q in b.evidence_quotes:
                        st.markdown(f"> {q}")
                else:
                    st.write("No evidence quotes available.")

            with right:
                st.markdown("**Support**")
                support_donut(b.support_score, b.support_label)

with tabs[1]:
    st.subheader("Baseline (non-grounded) summary")
    st.write("This summarises the whole extracted text directly (no retrieval). Useful for comparison.")

    if st.button("Generate baseline summary"):
        with st.spinner("Summarising…"):
            base = sys.baseline_summary(full_text)
        st.markdown("**Baseline output**")
        st.write(base)

with tabs[2]:
    st.subheader("Extracted text preview")
    st.write("This helps confirm extraction quality (e.g., if PDF is scanned images, extraction may be poor).")

    preview_pages = st.slider("Preview pages", 1, min(5, len(bundle["pages"])), 2)
    preview = "\n\n".join([txt for _, txt in bundle["pages"][:preview_pages]])
    st.text_area("Preview", value=preview[:8000], height=450)