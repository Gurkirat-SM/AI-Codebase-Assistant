"""Streamlit UI for the AI Codebase Assistant.

Run from the repository root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from backend.app.services.llm import GeminiService
from backend.app.services.retriever import RagService

st.set_page_config(page_title="AI Codebase Assistant", page_icon="🧠", layout="wide")

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNPACKED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000


def safe_extract_zip(archive_bytes: bytes, destination: Path) -> Path:
    """Extract a ZIP safely, rejecting traversal paths and oversized archives."""
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("The uploaded ZIP exceeds the 200 MB upload limit.")
    destination.mkdir(parents=True, exist_ok=True)
    total_unpacked = 0
    file_count = 0
    root = destination.resolve()

    with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError("The ZIP contains too many files.")
        for member in members:
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("The ZIP contains an unsafe file path.")
            target = (root / member_path).resolve()
            if target != root and root not in target.parents:
                raise ValueError("The ZIP contains an unsafe file path.")
            total_unpacked += member.file_size
            if total_unpacked > MAX_UNPACKED_BYTES:
                raise ValueError("The extracted archive would exceed the 500 MB limit.")
            if not member.is_dir():
                file_count += 1
        archive.extractall(root)

    if file_count == 0:
        raise ValueError("The ZIP archive is empty.")
    children = list(root.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return root


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets or the environment."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("GEMINI_API_KEY") or None


def initialize_services(index_dir: Path, api_key: str | None) -> tuple[RagService, GeminiService]:
    return RagService(storage_dir=index_dir), GeminiService(api_key=api_key)


st.title("🧠 AI Codebase Assistant")
st.caption("Upload a code repository, build a semantic index, and ask questions about its implementation.")

with st.sidebar:
    st.header("Configuration")
    api_key = get_api_key()
    if api_key:
        st.success("Gemini API key detected")
    else:
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Used only for generating answers. It is not displayed or saved by this app.",
        ) or None
        st.info("Add GEMINI_API_KEY to your environment or Streamlit secrets for persistent configuration.")

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a repository ZIP.\n"
        "2. Scan and chunk source files.\n"
        "3. Generate embeddings and build a FAISS index.\n"
        "4. Ask questions; Gemini answers using retrieved code context."
    )

for key, default in {
    "workspace": None,
    "rag_service": None,
    "gemini_service": None,
    "indexed": False,
    "index_summary": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

upload_col, action_col = st.columns([3, 1])
with upload_col:
    uploaded_zip = st.file_uploader("Upload your codebase (.zip)", type=["zip"])
with action_col:
    st.write("")
    st.write("")
    index_clicked = st.button(
        "Scan & index repository",
        type="primary",
        use_container_width=True,
        disabled=uploaded_zip is None,
    )

if index_clicked and uploaded_zip is not None:
    try:
        with st.spinner("Extracting repository and building the semantic index..."):
            workspace = tempfile.TemporaryDirectory(prefix="codebase-assistant-")
            workspace_path = Path(workspace.name)
            repository_path = safe_extract_zip(uploaded_zip.getvalue(), workspace_path / "repository")
            rag, gemini = initialize_services(workspace_path / "faiss-index", api_key)
            summary = rag.index_repository(repo_path=str(repository_path), persist=True)

            st.session_state.workspace = workspace
            st.session_state.rag_service = rag
            st.session_state.gemini_service = gemini
            st.session_state.indexed = summary.status == "success"
            st.session_state.index_summary = summary

        if st.session_state.indexed:
            st.success(summary.message)
        else:
            st.warning(summary.message)
    except zipfile.BadZipFile:
        st.error("That file is not a valid ZIP archive.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

summary = st.session_state.index_summary
if summary is not None:
    st.divider()
    st.subheader("Repository index")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Files scanned", summary.total_files)
    metric2.metric("Code chunks", summary.total_chunks)
    metric3.metric("Embedding dimensions", summary.embedding_dimension)
    st.caption(f"Embedding model: {summary.model_name}")

st.divider()
st.subheader("Ask your codebase")
if not st.session_state.indexed:
    st.info("Upload a ZIP and index it before asking questions.")
else:
    question = st.text_area(
        "What would you like to understand?",
        placeholder="Example: How does the repository scanner exclude binary files?",
        height=100,
    )
    top_k = st.slider("Number of code sections to retrieve", min_value=1, max_value=8, value=3)

    if st.button("Ask question", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching the codebase..."):
                retrieval = st.session_state.rag_service.search(question, top_k=top_k)
            chunks = [item.chunk for item in retrieval.results]

            st.markdown("### Answer")
            answer_area = st.empty()
            with st.spinner("Gemini is generating an answer..."):
                answer_area.write_stream(
                    st.session_state.gemini_service.generate_answer_stream(question, chunks)
                )

            st.markdown("### Retrieved source code")
            if not retrieval.sources:
                st.warning("No relevant code sections were retrieved.")
            for number, source in enumerate(retrieval.sources, start=1):
                heading = (
                    f"{number}. {source.file_path} · lines {source.start_line}–"
                    f"{source.end_line} · similarity {source.score:.3f}"
                )
                with st.expander(heading):
                    st.caption(source.language)
                    st.code(source.snippet, language=None)
        except Exception as exc:
            st.error(f"Could not answer this question: {exc}")

st.caption("Your uploaded repository is processed in this app session. The temporary workspace is not committed to GitHub.")
