"""Streamlit Web UI for the Local RAG Knowledge Base.

Launch with:
    streamlit run app.py
"""

import os
import time
import warnings

import streamlit as st

from src.config import load_config
from src.logging.logger import setup_logging

# Suppress warnings from transformers and other libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Local RAG Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ─────────────────────────────
def init_session_state() -> None:
    """Initialise session state variables on first load."""
    if "pipeline" not in st.session_state:
        # Show loading screen during initialization
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("### 🚀 Starting Local RAG System...")
            st.markdown("Initializing components, please wait...")
            
            # Progress bar for initialization steps
            progress_bar = st.progress(0, text="Setting up logging...")
            time.sleep(0.5)
            
            progress_bar.progress(20, text="Loading configuration...")
            setup_logging()
            config = load_config()
            time.sleep(0.5)
            
            progress_bar.progress(40, text="Initializing vector store...")
            time.sleep(0.5)
            
            progress_bar.progress(60, text="Loading LLM and embedding models...")
            time.sleep(0.5)
            
            progress_bar.progress(80, text="Setting up retrieval pipeline...")
            from src.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline(config)
            st.session_state.config = config
            time.sleep(0.5)
            
            progress_bar.progress(100, text="Ready! 🎉")
            time.sleep(0.5)
            
        # Clear the loading screen
        loading_placeholder.empty()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


init_session_state()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    config = st.session_state.config
    llm_model = config["llm"]["model"]
    emb_model = config["embedding"]["model"]
    doc_count = st.session_state.pipeline.doc_store.count
    code_count = st.session_state.pipeline.code_store.count
    total_count = doc_count + code_count
    retrieval_mode = "Hybrid (BM25 + RRF + Rerank)" if config.get(
        "retrieval", {}
    ).get("hybrid_mode", False) else "Dense Only"

    st.markdown("### System Info")
    st.markdown(f"- **LLM**: `{llm_model}`")
    st.markdown(f"- **Embedding**: `{emb_model}`")
    st.markdown(f"- **Documents**: `{doc_count}` docs / `{code_count}` code")
    st.markdown(f"- **Retrieval**: {retrieval_mode}")
    
    # Check reranker status
    reranker_status = "✅ Active"
    if config.get("retrieval", {}).get("hybrid_mode", False):
        if hasattr(st.session_state.pipeline, '_hybrid_retriever') and st.session_state.pipeline._hybrid_retriever:
            if not st.session_state.pipeline._hybrid_retriever._reranker:
                reranker_status = "⚠️ Disabled (connection failed)"
        else:
            reranker_status = "❌ Not initialized"
    
    st.markdown(f"- **Reranker**: {reranker_status}")

    st.divider()

    st.session_state.show_sources = st.toggle(
        "Show sources", value=st.session_state.show_sources
    )

    top_n = st.slider(
        "Top-N results",
        min_value=1,
        max_value=20,
        value=config.get("retrieval", {}).get("top_n", 5),
        help="Number of context chunks to feed to the LLM",
    )

    search_scope = st.radio(
        "Search scope",
        options=["all", "docs", "code"],
        format_func={"all": "All", "docs": "Documents only", "code": "Code only"}.get,
        horizontal=True,
        help="Filter which knowledge base to search",
    )

    # Prompt template selection (from PromptVersionManager)
    from src.generation.prompt_templates import list_available_templates
    available = list_available_templates(active_only=True)
    template_keys = [t["key"] for t in available]
    template_descs = {
        t["key"]: f"{t['key']} — {t.get('description', '')}"
        for t in available
    }

    template_name = st.selectbox(
        "Prompt template",
        options=template_keys,
        index=0,
        format_func=lambda k: template_descs.get(k, k),
    )

    # Prompt version details
    selected_info = next((t for t in available if t["key"] == template_name), None)
    if selected_info and selected_info.get("source") == "yaml":
        with st.expander("📝 Prompt details", expanded=False):
            st.markdown(f"- **Author**: {selected_info.get('author', 'unknown')}")
            st.markdown(f"- **Created**: {selected_info.get('created_at', '')}")
            st.markdown(f"- **Tags**: {', '.join(selected_info.get('tags', []))}")

            # Show version history for this template name
            from src.generation.prompt_version_manager import PromptVersionManager
            mgr = PromptVersionManager()
            tpl_name = selected_info.get("name", "")
            history = mgr.get_version_history(tpl_name)
            if len(history) > 1:
                st.markdown(f"**Versions** ({len(history)}):")
                for h in history:
                    marker = " ✅" if h["active"] else " ⛔"
                    st.caption(f"{h['key']}{marker} — {h.get('created_at', '')}")

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Document ingestion section
    st.markdown("### 📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload files to knowledge base",
        type=["txt", "md", "pdf", "pptx", "htm", "html"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Ingest", use_container_width=True):
        # Create a placeholder for ingestion progress
        ingestion_placeholder = st.empty()
        
        with ingestion_placeholder.container():
            st.markdown("### 📄 Processing Documents...")
            
            # Overall progress
            overall_progress = st.progress(0, text="Starting ingestion...")
            
            import os
            import tempfile
            from src.ingest import ingest_file
            from src.ingestion.chunker import DocumentChunker

            chunker = DocumentChunker(config)
            vs = st.session_state.pipeline.vector_store
            total_added = 0
            
            for i, uf in enumerate(uploaded_files):
                # Update progress for current file
                file_progress = (i / len(uploaded_files)) * 100
                overall_progress.progress(int(file_progress), text=f"Processing {uf.name}...")
                
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name

                try:
                    # Show file-specific progress
                    with st.spinner(f"Parsing {uf.name}..."):
                        chunks = ingest_file(
                            tmp_path, chunker, config=config
                        )

                    # Preserve original filename in chunk metadata
                    # (temp file gets a random name like tmp12345.pdf)
                    for chunk in chunks:
                        meta = chunk.get("metadata", {})
                        meta["file_name"] = uf.name
                        meta["source_file"] = uf.name
                    
                    if chunks:
                        with st.spinner(f"Adding {len(chunks)} chunks to vector store..."):
                            added = vs.add_chunks(chunks)
                            total_added += added
                            st.success(f"✅ {uf.name}: {added} chunks added")
                    else:
                        st.warning(f"⚠️ {uf.name}: No content extracted")
                except Exception as e:
                    st.error(f"❌ {uf.name}: Error processing - {str(e)}")
                finally:
                    os.unlink(tmp_path)

            # Rebuilding BM25 index
            overall_progress.progress(90, text="Rebuilding search index...")
            with st.spinner("Rebuilding BM25 index for hybrid search..."):
                st.session_state.pipeline.rebuild_bm25_index()
            
            overall_progress.progress(100, text="Complete! 🎉")
            st.success(f"✅ Ingestion complete: {total_added} chunks from {len(uploaded_files)} file(s)")
            
            # Keep the success message visible for a moment
            time.sleep(2)
            
        # Clear the ingestion placeholder
        ingestion_placeholder.empty()
        
        # Show final success message in the sidebar
        st.success(f"Added {total_added} new chunks from {len(uploaded_files)} file(s)")
        st.rerun()

    st.divider()

    # Code ingestion section
    st.markdown("### 💻 Ingest Code")
    code_source = st.radio(
        "Source type",
        options=["Local Folder", "GitHub URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if code_source == "Local Folder":
        code_path = st.text_input(
            "Folder path",
            placeholder=r"C:\path\to\your\project",
            help="Absolute path to a local code folder",
        )
    else:
        code_path = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repo",
            help="Public Git repository URL (will be shallow-cloned)",
        )

    code_repo_name = st.text_input(
        "Repository name (optional)",
        placeholder="Auto-detected from path",
        help="Custom name to identify this codebase in search results",
    )

    if code_path and st.button("📥 Ingest Code", use_container_width=True):
        code_ingest_placeholder = st.empty()

        with code_ingest_placeholder.container():
            st.markdown("### 💻 Processing Code...")
            code_progress = st.progress(0, text="Discovering code files...")

            from src.ingest import run_code_ingestion

            try:
                code_vs = st.session_state.pipeline.code_store
                repo = code_repo_name.strip() if code_repo_name.strip() else None

                code_progress.progress(20, text="Parsing and chunking code...")
                file_count, chunk_count, added_count = run_code_ingestion(
                    path=code_path.strip(),
                    repo_name=repo,
                    config=config,
                    vector_store=code_vs,
                )

                code_progress.progress(60, text="Rebuilding search index...")
                with st.spinner("Rebuilding BM25 index..."):
                    st.session_state.pipeline.rebuild_bm25_index()

                code_progress.progress(80, text="Building repo map...")
                with st.spinner("Building RepoMap (symbol graph + PageRank)..."):
                    st.session_state.pipeline.build_repo_map()
                    # 记录构建时间
                    from datetime import datetime
                    st.session_state["repo_map_last_build"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                code_progress.progress(100, text="Complete! 🎉")
                st.success(
                    f"✅ Code ingestion complete: {file_count} files, "
                    f"{chunk_count} chunks, {added_count} new"
                )
                time.sleep(2)

            except FileNotFoundError as e:
                st.error(f"❌ Path not found: {e}")
            except RuntimeError as e:
                st.error(f"❌ Error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

        code_ingest_placeholder.empty()
        st.rerun()

    # ── RepoMap controls ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ RepoMap")

    repo_map_text = st.session_state.pipeline.repo_map_text
    rm = st.session_state.pipeline.repo_map

    # Status display and statistics
    if rm:
        # Get last build time from session state
        last_build_time = st.session_state.get("repo_map_last_build", "Unknown")
        
        st.markdown(
            f"**{rm.file_count}** files · "
            f"**{rm.definition_count}** defs · "
            f"**{rm.reference_count}** refs"
        )
        st.caption(f"🕒 Last built: {last_build_time}")
        
        # Build status
        st.success("✅ RepoMap built")
        button_text = "🔄 Rebuild RepoMap"
        help_text = "Rebuild RepoMap (to fix corruption or update symbol relationships)"
    else:
        st.caption("⚠️ RepoMap not built")
        button_text = "🔄 Build RepoMap"
        help_text = "Initial RepoMap build (requires code ingestion first)"

    # Build button
    if st.button(button_text, use_container_width=True, help=help_text):
        with st.spinner("Building RepoMap (tree-sitter + PageRank)..."):
            st.session_state.pipeline.build_repo_map()
            # Record build time
            from datetime import datetime
            st.session_state["repo_map_last_build"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

    # Help text
    if rm:
        st.markdown("""
        <div style="font-size: 0.85em; color: #666; margin-top: 0.5rem;">
            <strong>💡 About RepoMap:</strong>
            <ul style="margin: 0.25rem 0; padding-left: 1.2rem;">
                <li>Analyzes code structure, extracting classes, functions, and their relationships</li>
                <li>Uses PageRank algorithm to rank symbol importance for better LLM understanding</li>
                <li>Automatically built after code ingestion, usually no manual action needed</li>
                <li>Click "Rebuild" to fix corruption or update symbol relationships</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.85em; color: #666; margin-top: 0.5rem;">
            <strong>💡 About RepoMap:</strong>
            <ul style="margin: 0.25rem 0; padding-left: 1.2rem;">
                <li>Analyzes code structure, extracting classes, functions, and their relationships</li>
                <li>Uses PageRank algorithm to rank symbol importance for better LLM understanding</li>
                <li>Please ingest code files first using the "💻 Ingest Code" section above</li>
                <li>RepoMap is automatically built after code ingestion, usually no manual action needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if repo_map_text:
        with st.expander("📋 View RepoMap Details", expanded=False):
            st.code(repo_map_text, language="markdown")


# ── Main content ─────────────────────────────────────────────
st.title("📚 Local RAG Knowledge Base")
st.caption(f"Powered by Ollama (`{llm_model}`) · {doc_count} docs + {code_count} code indexed")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg and st.session_state.show_sources:
            with st.expander(
                f"📋 Sources ({len(msg['sources'])}) · "
                f"{msg.get('latency_ms', 0)}ms · "
                f"{msg.get('retrieval_mode', 'unknown')}",
                expanded=False,
            ):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score", 0)
                    ctype = src.get("content_type", "unknown")
                    st.markdown(
                        f"**[{i}]** `{src['file']}` [{ctype}] — score: `{score:.4f}`"
                    )
                    if src.get("content_preview"):
                        st.caption(src["content_preview"][:300])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with detailed progress
    with st.chat_message("assistant"):
        # Create a placeholder for query progress
        query_placeholder = st.empty()
        
        with query_placeholder.container():
            st.markdown("🔍 **Searching your knowledge base...**")
            
            # Step 1: Retrieval
            with st.spinner("🔎 Retrieving relevant documents..."):
                result = st.session_state.pipeline.query(
                    question=prompt,
                    top_n=top_n,
                    template_name=template_name,
                    search_scope=search_scope,
                )
            
            # Step 2: Generation (if we found sources)
            if result["sources"]:
                with st.spinner("🤖 Generating answer with LLM..."):
                    time.sleep(0.5)  # Brief pause to show the spinner
            else:
                st.warning("⚠️ No relevant documents found")
        
        # Clear the progress indicator and show the answer
        query_placeholder.empty()
        st.markdown(result["answer"])

        # Show sources
        if st.session_state.show_sources and result["sources"]:
            with st.expander(
                f"📋 Sources ({len(result['sources'])}) · "
                f"{result['latency_ms']}ms · "
                f"{result['retrieval_mode']}",
                expanded=False,
            ):
                for i, src in enumerate(result["sources"], 1):
                    score = src.get("score", 0)
                    ctype = src.get("content_type", "unknown")
                    st.markdown(
                        f"**[{i}]** `{src['file']}` [{ctype}] — score: `{score:.4f}`"
                    )
                    if src.get("content_preview"):
                        st.caption(src["content_preview"][:300])

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
        "retrieval_mode": result["retrieval_mode"],
        "trace_id": result["trace_id"],
    })
