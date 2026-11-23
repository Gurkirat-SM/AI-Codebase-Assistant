"""Streamlit UI for the AI Codebase Assistant.
A modern, professional developer dashboard for semantic codebase understanding.

Run from repository root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from backend.app.core.config import get_settings
from backend.app.services.llm import GeminiService
from backend.app.services.retriever import RagService
from backend.app.services.weighting import classify_file_type

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Codebase Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom developer-dashboard CSS
CUSTOM_CSS = """
<style>
/* Main typography and dark palette */
:root {
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --bg-card: #1e293b;
    --border-card: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

/* Header branding */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 8px;
    margin-bottom: 12px;
    border-bottom: 1px solid #334155;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-tagline {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.15s ease;
}
.metric-card:hover {
    border-color: #6366f1;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #6366f1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-code { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
.badge-test { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
.badge-config { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
.badge-doc { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
.badge-score { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.4); }

/* Latency pills */
.latency-bar {
    font-size: 0.75rem;
    color: #94a3b8;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    margin-bottom: 8px;
}

/* Prompt example chips */
.prompt-chip-btn {
    margin: 4px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Archive limits
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNPACKED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000

EXAMPLE_QUESTIONS = [
    ("🏛️ Architecture Overview", "Explain the high-level architecture of this project and verify against source code."),
    ("🔍 Scanner Implementation", "How does RepoScanner identify files and exclude binary or ignored paths? Cite line numbers."),
    ("⚡ Indexing Pipeline", "How does the indexing pipeline chunk, embed, and store code in FAISS and BM25?"),
    ("🌊 User Query Flow", "Trace the execution flow of a user query from search to Gemini answer generation."),
    ("🌐 API Routes", "Which files define API endpoints and how do they handle errors?"),
    ("🚀 Performance Areas", "Identify possible areas for performance optimization in retrieval and caching."),
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def safe_extract_zip(archive_bytes: bytes, destination: Path) -> Path:
    """Extract a ZIP archive safely, rejecting directory traversals and oversized archives."""
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("The uploaded ZIP exceeds the 200 MB upload limit.")
    destination.mkdir(parents=True, exist_ok=True)
    total_unpacked = 0
    file_count = 0
    root = destination.resolve()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
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
                raise ValueError("The extracted archive exceeds the 500 MB limit.")
            if not member.is_dir():
                file_count += 1
        archive.extractall(root)

    if file_count == 0:
        raise ValueError("The ZIP archive contains no files.")
    children = list(root.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return root


def get_api_key() -> str | None:
    """Read Gemini API key from Streamlit secrets, session state, or environment."""
    if st.session_state.get("custom_api_key"):
        return st.session_state.custom_api_key.strip()
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("GEMINI_API_KEY") or None


def initialize_services(index_dir: Path, api_key: str | None) -> tuple[RagService, GeminiService]:
    """Initialize session-isolated RagService and GeminiService."""
    return RagService(storage_dir=index_dir), GeminiService(api_key=api_key)


def get_category_badge(file_path: str) -> str:
    """Return an HTML badge string for the file category."""
    cat = classify_file_type(file_path)
    if cat == "code":
        return '<span class="badge badge-code">Code</span>'
    elif cat == "test":
        return '<span class="badge badge-test">Test</span>'
    elif cat == "configuration":
        return '<span class="badge badge-config">Config</span>'
    else:
        return '<span class="badge badge-doc">Docs</span>'


def reset_session():
    """Clean up temp directory and reset all session state variables."""
    if st.session_state.get("workspace"):
        try:
            st.session_state.workspace.cleanup()
        except Exception:
            pass
    st.session_state.workspace = None
    st.session_state.rag_service = None
    st.session_state.gemini_service = None
    st.session_state.indexed = False
    st.session_state.index_summary = None
    st.session_state.repo_name = None
    st.session_state.messages = []
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
defaults = {
    "workspace": None,
    "rag_service": None,
    "gemini_service": None,
    "indexed": False,
    "index_summary": None,
    "repo_name": None,
    "messages": [],
    "custom_api_key": "",
    "pending_prompt": None,
    "feedback_state": {},
    # Retrieval configuration
    "cfg_top_k": 4,
    "cfg_enable_hybrid": True,
    "cfg_enable_reranker": True,
    "cfg_enable_weighting": True,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar Redesign
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo & App Title
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 2rem;">🧠</span>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Codebase Assistant</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Hybrid RAG Developer Tool</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation Radio
    nav_selection = st.radio(
        "Navigation",
        options=["💬 Codebase Chat", "📊 Repository Dashboard", "📁 Code Explorer", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()

    # Repository Upload Section
    st.markdown("#### 📦 Repository")
    uploaded_zip = st.file_uploader(
        "Upload repository (.zip)",
        type=["zip"],
        help="Upload a ZIP of your codebase. Maximum size 200 MB.",
    )

    # Repository status badge
    if st.session_state.indexed and st.session_state.index_summary:
        summary = st.session_state.index_summary
        st.markdown(
            f"""
            <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 6px; padding: 10px; margin-top: 6px;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="color: #22c55e;">●</span>
                    <strong style="color: #22c55e; font-size: 0.85rem;">Indexed</strong>
                </div>
                <div style="font-size: 0.8rem; color: #e2e8f0; margin-top: 4px; word-break: break-all;">
                    📁 <strong>{st.session_state.repo_name or 'Repository'}</strong>
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                    {summary.total_files} files · {summary.total_chunks} chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 6px; padding: 10px; margin-top: 6px;">
                <span style="color: #94a3b8;">○</span>
                <span style="color: #94a3b8; font-size: 0.82rem; margin-left: 4px;">No repository indexed</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Index Button (only active if file uploaded and not already indexed)
    index_btn = st.button(
        "⚡ Index Repository",
        type="primary",
        use_container_width=True,
        disabled=uploaded_zip is None,
    )

    # API Key Configuration
    st.markdown("#### 🔑 Gemini API")
    current_key = get_api_key()
    if current_key:
        masked = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "Configured"
        st.markdown(
            f"""
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 6px; padding: 8px 10px; font-size: 0.8rem; color: #93c5fd;">
                ✓ API Key Connected ({masked})
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        new_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Set in environment as GEMINI_API_KEY or enter here. Stored in session only.",
        )
        if new_key:
            st.session_state.custom_api_key = new_key.strip()
            st.rerun()

    st.divider()

    # Session Reset Button
    if st.session_state.indexed or st.session_state.messages:
        if st.button("🔄 Reset Current Session", use_container_width=True):
            reset_session()
            st.rerun()


# ---------------------------------------------------------------------------
# Repository Indexing Workflow
# ---------------------------------------------------------------------------
if index_btn and uploaded_zip is not None:
    try:
        progress_bar = st.progress(0, text="Initializing workspace...")
        with st.spinner("Extracting repository archive..."):
            workspace = tempfile.TemporaryDirectory(prefix="codebase-assistant-")
            workspace_path = Path(workspace.name)
            progress_bar.progress(25, text="Extracting and verifying ZIP files...")
            repository_path = safe_extract_zip(uploaded_zip.getvalue(), workspace_path / "repository")

            progress_bar.progress(50, text="Initializing RAG and Gemini services...")
            rag, gemini = initialize_services(workspace_path / "index", get_api_key())

            progress_bar.progress(75, text="Parsing code, building FAISS & BM25 indices...")
            summary = rag.index_repository(repo_path=str(repository_path), persist=True)

            progress_bar.progress(100, text="Index complete!")

            st.session_state.workspace = workspace
            st.session_state.rag_service = rag
            st.session_state.gemini_service = gemini
            st.session_state.indexed = (summary.status == "success")
            st.session_state.index_summary = summary
            st.session_state.repo_name = uploaded_zip.name.replace(".zip", "")
            st.session_state.messages = []

        if st.session_state.indexed:
            st.success(f"✓ Indexed {summary.total_chunks} chunks across {summary.total_files} files successfully!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning(summary.message)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")


# ---------------------------------------------------------------------------
# VIEW 1: Codebase Chat
# ---------------------------------------------------------------------------
if nav_selection == "💬 Codebase Chat":
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-title">💬 Interactive Codebase Chat</h1>
                <p class="app-tagline">Ask questions about architecture, algorithms, and implementations with grounded line-level citations.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.indexed:
        st.info("👋 Upload a repository ZIP in the sidebar and click **Index Repository** to begin asking questions.")

        # Show preview of suggested questions
        st.markdown("##### 💡 Example Questions Supported Once Indexed:")
        col1, col2 = st.columns(2)
        for i, (title, q_text) in enumerate(EXAMPLE_QUESTIONS):
            target_col = col1 if i % 2 == 0 else col2
            target_col.markdown(
                f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                    <strong style="color: #6366f1; font-size: 0.85rem;">{title}</strong>
                    <p style="margin: 4px 0 0; font-size: 0.8rem; color: #94a3b8;">{q_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Chat Controls Row (Clear Chat, Export)
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([6, 2, 2])
        with ctrl_col2:
            retrieval_k = st.selectbox(
                "Sections to retrieve",
                options=[2, 3, 4, 5, 6, 8],
                index=2,
                help="Number of top code chunks passed to Gemini.",
                key="chat_top_k",
            )
        with ctrl_col3:
            st.write("")
            if st.button("🗑️ Clear Chat", use_container_width=True, disabled=len(st.session_state.messages) == 0):
                st.session_state.messages = []
                st.rerun()

        # Suggestion chips when chat is empty
        if len(st.session_state.messages) == 0:
            st.markdown("##### 💡 Suggested Questions")
            chip_cols = st.columns(3)
            for i, (title, q_text) in enumerate(EXAMPLE_QUESTIONS[:6]):
                with chip_cols[i % 3]:
                    if st.button(title, use_container_width=True, key=f"chip_{i}"):
                        st.session_state.pending_prompt = q_text
                        st.rerun()

        # Render conversation history
        for msg_idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

                # Render sources and latencies for assistant responses
                if role == "assistant":
                    sources = msg.get("sources", [])
                    latencies = msg.get("latencies", {})

                    # Latency metrics badge
                    if latencies:
                        retrieval_t = latencies.get("total_retrieval_sec", 0.0)
                        gen_t = latencies.get("llm_generation_sec", 0.0)
                        rerank_t = latencies.get("reranking_sec", 0.0)
                        st.markdown(
                            f"""
                            <div class="latency-bar">
                                <span>⚡ <strong>Retrieval:</strong> {retrieval_t:.3f}s</span>
                                <span>·</span>
                                <span>🎯 <strong>Rerank:</strong> {rerank_t:.3f}s</span>
                                <span>·</span>
                                <span>🤖 <strong>Generation:</strong> {gen_t:.3f}s</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Collapsible sources section
                    if sources:
                        with st.expander(f"📑 Retrieved Source Sections ({len(sources)})", expanded=False):
                            for s_idx, src in enumerate(sources, 1):
                                score_pct = round(src.get("score", 0.0) * 100, 1)
                                cat_badge = get_category_badge(src["file_path"])
                                heading_html = (
                                    f"<div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;'>"
                                    f"<div><strong>#{s_idx}</strong> <code>{src['file_path']}</code> (lines {src['start_line']}–{src['end_line']})</div>"
                                    f"<div>{cat_badge} <span class='badge badge-score'>{score_pct}% Match</span></div>"
                                    f"</div>"
                                )
                                st.markdown(heading_html, unsafe_allow_html=True)
                                st.code(src["snippet"], language=src.get("language", "").lower() or None)

                    # Feedback buttons
                    fb_col1, fb_col2, _ = st.columns([1, 1, 10])
                    with fb_col1:
                        if st.button("👍", key=f"fb_up_{msg_idx}", help="Helpful response"):
                            st.session_state.feedback_state[msg_idx] = "up"
                            st.toast("Thank you for your feedback!")
                    with fb_col2:
                        if st.button("👎", key=f"fb_down_{msg_idx}", help="Needs improvement"):
                            st.session_state.feedback_state[msg_idx] = "down"
                            st.toast("Feedback recorded.")

        # Chat Input Bar
        chat_prompt = st.chat_input("Ask a question about this repository...")
        prompt_to_run = chat_prompt or st.session_state.pending_prompt

        if prompt_to_run:
            st.session_state.pending_prompt = None

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            # Generate Assistant response
            with st.chat_message("assistant"):
                rag_svc: RagService = st.session_state.rag_service
                gemini_svc: GeminiService = st.session_state.gemini_service

                t_retrieval_start = time.perf_counter()
                with st.spinner("Searching codebase with hybrid vector & BM25 retrieval..."):
                    retrieval_res = rag_svc.search(
                        query=prompt_to_run,
                        top_k=retrieval_k,
                        enable_hybrid=st.session_state.cfg_enable_hybrid,
                        enable_reranker=st.session_state.cfg_enable_reranker,
                    )
                t_retrieval_end = time.perf_counter()

                retrieved_chunks = [r.chunk for r in retrieval_res.results]
                sources_payload = [
                    {
                        "file_path": s.file_path,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "score": s.score,
                        "language": s.language,
                        "snippet": s.snippet,
                    }
                    for s in retrieval_res.sources
                ]

                # Stream answer from Gemini
                t_gen_start = time.perf_counter()
                answer_area = st.empty()
                with st.spinner("Gemini is analyzing grounded code context..."):
                    if hasattr(gemini_svc, "generate_answer_stream"):
                        full_answer = answer_area.write_stream(
                            gemini_svc.generate_answer_stream(prompt_to_run, retrieved_chunks)
                        )
                    else:
                        full_answer = gemini_svc.generate_answer(prompt_to_run, retrieved_chunks)
                        answer_area.markdown(full_answer)
                t_gen_end = time.perf_counter()

                # Calculate latency breakdown
                latencies = dict(retrieval_res.latencies or {})
                latencies["total_retrieval_sec"] = round(t_retrieval_end - t_retrieval_start, 4)
                latencies["llm_generation_sec"] = round(t_gen_end - t_gen_start, 4)

                # Append assistant message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "sources": sources_payload,
                        "latencies": latencies,
                    }
                )
                st.rerun()


# ---------------------------------------------------------------------------
# VIEW 2: Repository Dashboard
# ---------------------------------------------------------------------------
elif nav_selection == "📊 Repository Dashboard":
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-title">📊 Repository Dashboard</h1>
                <p class="app-tagline">Detailed structural analytics and indexing statistics of the loaded codebase.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.indexed or not st.session_state.index_summary:
        st.info("No repository indexed. Please upload and index a codebase in the sidebar to view metrics.")
    else:
        summary = st.session_state.index_summary
        rag_svc: RagService = st.session_state.rag_service
        scan_summary = getattr(rag_svc, "last_scan_summary", None)

        # Primary Metric Cards Row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary.total_files}</div>
                    <div class="metric-label">Files Scanned</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary.total_chunks}</div>
                    <div class="metric-label">Code Chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            total_lines = scan_summary.total_lines if scan_summary else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{total_lines}</div>
                    <div class="metric-label">Total Lines of Code</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            lang_count = len(scan_summary.files_by_language) if scan_summary else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{lang_count}</div>
                    <div class="metric-label">Languages Detected</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Languages Distribution & Index Metadata
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("##### 📈 Programming Language Breakdown")
            if scan_summary and scan_summary.files_by_language:
                lang_data = scan_summary.files_by_language
                st.bar_chart(lang_data)
            else:
                st.write("Language distribution data unavailable.")

        with col_right:
            st.markdown("##### ⚙️ Indexing & Vector Specifications")
            st.markdown(
                f"""
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; font-size: 0.85rem; line-height: 1.8;">
                    <div><strong>Repository Path:</strong> <code style="word-break: break-all;">{summary.repo_path}</code></div>
                    <div><strong>Embedding Model:</strong> <code>{summary.model_name}</code></div>
                    <div><strong>Vector Dimensions:</strong> <code>{summary.embedding_dimension}</code> (Dense FAISS IP)</div>
                    <div><strong>Keyword Search:</strong> <code>BM25Okapi</code> (Code-Aware Tokenizer)</div>
                    <div><strong>Reranker Model:</strong> <code>ms-marco-MiniLM-L-6-v2</code> (Cross-Encoder)</div>
                    <div><strong>Chunking Engine:</strong> AST Python Parser + Sliding Window Fallback</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# VIEW 3: Code Explorer
# ---------------------------------------------------------------------------
elif nav_selection == "📁 Code Explorer":
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-title">📁 Codebase File Explorer</h1>
                <p class="app-tagline">Browse, search, and inspect the scanned files and line counts.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.indexed or not st.session_state.rag_service:
        st.info("No repository indexed. Please upload a repository in the sidebar.")
    else:
        rag_svc: RagService = st.session_state.rag_service
        scan_summary = getattr(rag_svc, "last_scan_summary", None)

        if not scan_summary or not scan_summary.files:
            st.warning("File metadata unavailable for the current index.")
        else:
            files_list = scan_summary.files
            filter_col1, filter_col2 = st.columns([3, 1])
            with filter_col1:
                search_term = st.text_input("Filter by file path", placeholder="e.g. services, scanner, test")
            with filter_col2:
                available_langs = ["All"] + sorted(list(scan_summary.files_by_language.keys()))
                lang_filter = st.selectbox("Language filter", options=available_langs)

            # Filter files
            filtered_files = files_list
            if search_term.strip():
                filtered_files = [f for f in filtered_files if search_term.lower() in f.relative_path.lower()]
            if lang_filter != "All":
                filtered_files = [f for f in filtered_files if f.language == lang_filter]

            st.markdown(f"**Showing {len(filtered_files)} of {len(files_list)} files**")

            # Render file table
            table_data = [
                {
                    "File Path": f.relative_path,
                    "Category": classify_file_type(f.relative_path).capitalize(),
                    "Language": f.language,
                    "Lines": f.line_count,
                    "Size (bytes)": f.size_bytes,
                }
                for f in filtered_files
            ]
            st.dataframe(table_data, use_container_width=True, height=450)


# ---------------------------------------------------------------------------
# VIEW 4: Settings & Configuration
# ---------------------------------------------------------------------------
elif nav_selection == "⚙️ Settings":
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1 class="app-title">⚙️ Retrieval & Engine Settings</h1>
                <p class="app-tagline">Fine-tune the hybrid search pipeline, reranking weights, and model parameters.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    settings = get_settings()

    st.markdown("##### 🎛️ Retrieval Pipeline Settings")
    st.session_state.cfg_enable_hybrid = st.toggle(
        "Enable Hybrid Search (FAISS Dense Vector + BM25 Keyword Search)",
        value=st.session_state.cfg_enable_hybrid,
        help="Combines dense semantic search with sparse exact keyword search using Reciprocal Rank Fusion.",
    )

    st.session_state.cfg_enable_reranker = st.toggle(
        "Enable Cross-Encoder Reranker",
        value=st.session_state.cfg_enable_reranker,
        help="Scores candidate chunks with cross-encoder/ms-marco-MiniLM-L-6-v2 for highest precision.",
    )

    st.session_state.cfg_enable_weighting = st.toggle(
        "Enable Query-Dependent File Weighting",
        value=st.session_state.cfg_enable_weighting,
        help="Dynamically prioritizes test, configuration, documentation, or code implementation files based on user query intent.",
    )

    st.session_state.cfg_top_k = st.slider(
        "Default Code Chunks Retrieved (top_k)",
        min_value=1,
        max_value=10,
        value=st.session_state.cfg_top_k,
        help="Controls how many candidate code chunks are sent to Gemini as context.",
    )

    st.divider()
    st.markdown("##### ℹ️ Active Engine Configuration")
    st.markdown(
        f"""
        - **Embedding Model:** `{settings.EMBEDDING_MODEL_NAME}`
        - **Vector Index Type:** `FAISS IndexFlatIP (Inner Product)`
        - **BM25 Parameters:** `k1=1.5, b=0.75 (Lucene Positive IDF)`
        - **Cross-Encoder Model:** `{settings.RERANKER_MODEL_NAME}`
        - **LLM Model:** `{settings.GEMINI_MODEL}`
        - **Max Context Budget:** `{settings.MAX_CONTEXT_CHARS:,} characters`
        """
    )
