"""
app.py - CodeSense AI — Main Streamlit Application
No API key input, no demo mode, no token sliders on the UI.
Everything is hardcoded and clean.
"""

import streamlit as st
from llm_handler import LLMHandler
from rag_pipeline import RAGPipeline
from utils import detect_language

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CodeSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d14 !important;
    color: #c9d1e0 !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMain"] { background: #0a0d14 !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

/* ── Hero ── */
.hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
    border: 1px solid #1e2d40;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    letter-spacing: -0.5px;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub  { color: #6b7a99; font-size: 1rem; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; }
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 8px;
    margin-top: 12px;
}

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d4ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
}

/* ── Textarea ── */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 10px !important;
    color: #c9d1e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0d1117, #161b27) !important;
    border: 1px solid #1e2d40 !important;
    color: #c9d1e0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.18) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Output card ── */
.output-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin: 1rem 0;
    position: relative;
    animation: fadeUp 0.35s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.output-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
}
.output-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d4ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.output-content {
    color: #c9d1e0;
    font-size: 0.94rem;
    line-height: 1.85;
    white-space: pre-wrap;
}

/* ── Chat messages ── */
.chat-msg {
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.7;
    animation: fadeUp 0.3s ease;
}
.chat-msg.user {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.18);
    border-left: 3px solid #00d4ff;
}
.chat-msg.assistant {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.18);
    border-left: 3px solid #7c3aed;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.87rem;
}
.chat-label { font-family: 'JetBrains Mono', monospace; font-size: 0.67rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; }
.chat-label.user { color: #00d4ff; }
.chat-label.bot  { color: #7c3aed; }

/* ── Language chip ── */
.lang-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3);
    color: #a78bfa; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 3px 12px; border-radius: 20px; margin-top: 0.5rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important; border: 1px solid #1e2d40 !important;
    border-radius: 8px !important; color: #c9d1e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px dashed #1e2d40 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,212,255,0.4) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d40 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.sidebar-logo {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem; text-align: center;
}
.sidebar-tagline {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    color: #3d4a5c; text-align: center; margin-bottom: 1.5rem;
}
.sidebar-card {
    background: #161b27; border: 1px solid #1e2d40;
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
}
.sidebar-card-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: #6b7a99; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #1e2d40;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: #6b7a99; }
.info-value { color: #00d4ff; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    display: inline-block; margin-right: 6px;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { background: #0d1117 !important; border-bottom: 1px solid #1e2d40 !important; }
[data-testid="stTabs"] [role="tab"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; color: #6b7a99 !important; border: none !important; padding: 0.6rem 1.4rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff !important; background: transparent !important; }

/* ── Misc ── */
hr { border-color: #1e2d40 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff33; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_code"    not in st.session_state: st.session_state.last_code    = ""
if "rag"          not in st.session_state: st.session_state.rag          = RAGPipeline()
if "llm"          not in st.session_state: st.session_state.llm          = LLMHandler()


# ─────────────────────────────────────────────
# SIDEBAR — Info only, no settings inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ CodeSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">// AI-Powered Code Understanding</div>', unsafe_allow_html=True)

    # Status card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">System Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-row">
        <span class="info-label"><span class="status-dot"></span>LLM Engine</span>
        <span class="info-value">Online</span>
    </div>
    <div class="info-row">
        <span class="info-label">Model</span>
        <span class="info-value">Llama 3.3 70B</span>
    </div>
    <div class="info-row">
        <span class="info-label">Provider</span>
        <span class="info-value">Groq</span>
    </div>
    <div class="info-row">
        <span class="info-label">RAG</span>
        <span class="info-value">FAISS + Embeddings</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Features card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#c9d1e0;line-height:2;">
        🔍 &nbsp;Code Explanation<br>
        📝 &nbsp;Summarization<br>
        🐛 &nbsp;Bug Detection<br>
        ⚡ &nbsp;Optimization Tips<br>
        💬 &nbsp;Interactive Q&A<br>
        🌍 &nbsp;Multi-Language Support<br>
        🧠 &nbsp;RAG Context Retrieval<br>
        📂 &nbsp;File Upload (.py .js .java .cpp)
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Supported languages card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">Supported Languages</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#a78bfa;line-height:2.1;">
        Python &nbsp;·&nbsp; JavaScript &nbsp;·&nbsp; Java<br>
        C++ &nbsp;·&nbsp; TypeScript &nbsp;·&nbsp; Go<br>
        Ruby &nbsp;·&nbsp; Rust &nbsp;·&nbsp; C
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️  Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:#2a3347;text-align:center;line-height:1.8;">'
        'CodeSense AI · v1.0<br>Academic Research Project<br>LLM + RAG Pipeline</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">⬡ CodeSense AI</div>
  <div class="hero-sub"></div>
  <div style="margin-top:14px;">
    <span class="badge">LLM Powered</span>
    <span class="badge">RAG Pipeline</span>
    <span class="badge">FAISS Vector Search</span>
    <span class="badge">Multi-Language</span>
    <span class="badge">Interactive Q&A</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LAYOUT — Input | Output
# ─────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

# ── LEFT: Code Input ─────────────────────────
with col_input:
    st.markdown('<div class="section-label">01 — Code Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a source file",
        type=["py", "java", "cpp", "js", "ts", "c", "go", "rb"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_code  = uploaded_file.read().decode("utf-8", errors="replace")
        code_input = st.text_area("code_area", value=file_code, height=340, label_visibility="collapsed")
    else:
        code_input = st.text_area(
            "code_area",
            placeholder="# Paste your code here...\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))",
            height=340,
            label_visibility="collapsed",
        )

    # Language selector
    languages     = ["Auto Detect","Python","JavaScript","Java","C++","TypeScript","Go","Ruby","Rust","C"]
    lang_col, _   = st.columns([1, 1])
    with lang_col:
        selected_lang = st.selectbox("Language", languages, label_visibility="collapsed")

    # Detect language
    if code_input.strip():
        if selected_lang == "Auto Detect":
            detected = detect_language(code_input)
        else:
            detected = selected_lang
        st.markdown(f'<div class="lang-chip">⬡ Detected: {detected}</div>', unsafe_allow_html=True)
    else:
        detected = "Unknown"

    st.markdown("<br>", unsafe_allow_html=True)

    # Action buttons
    b1, b2, b3, b4 = st.columns(4)
    explain_btn   = b1.button("🔍 Explain")
    summarize_btn = b2.button("📝 Summarize")
    debug_btn     = b3.button("🐛 Debug")
    optimize_btn  = b4.button("⚡ Optimize")


# ── RIGHT: Output Tabs ───────────────────────
with col_output:
    st.markdown('<div class="section-label">02 — Analysis Output</div>', unsafe_allow_html=True)

    tab_explain, tab_summary, tab_debug, tab_optimize = st.tabs([
        "🔍 Explanation", "📝 Summary", "🐛 Debug", "⚡ Optimize"
    ])

    def run_analysis(action_label: str, tab):
        """Index code into RAG, call LLM, display result."""
        if not code_input.strip():
            tab.warning("⚠️  Please paste or upload some code first.")
            return

        # Index code into FAISS
        st.session_state.rag.index_code(code_input, detected, chunk_size=256)

        with tab:
            with st.spinner(f"Analysing code — {action_label}..."):
                # Retrieve RAG context
                rag_context = st.session_state.rag.retrieve(
                    query = action_label + " " + code_input[:300],
                    top_k = 3,
                )
                # Call LLM
                result = st.session_state.llm.run(
                    code     = code_input,
                    language = detected,
                    action   = action_label,
                    context  = rag_context,
                )

            st.markdown(
                f'<div class="output-card">'
                f'<div class="output-card-title">◈ {action_label}</div>'
                f'<div class="output-content">{result}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Save to chat memory
        st.session_state.last_code = code_input
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"[{action_label}]\n{result}",
        })

    # Trigger correct action
    if explain_btn:
        with tab_explain:
            run_analysis("Explanation", tab_explain)
    if summarize_btn:
        with tab_summary:
            run_analysis("Summarization", tab_summary)
    if debug_btn:
        with tab_debug:
            run_analysis("Debugging", tab_debug)
    if optimize_btn:
        with tab_optimize:
            run_analysis("Optimization", tab_optimize)


# ─────────────────────────────────────────────
# CHAT / Q&A SECTION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">03 — Interactive Q&A with Your Code</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-msg user"><div class="chat-label user">YOU</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-msg assistant"><div class="chat-label bot">CODESENSE AI</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# Chat input row
chat_col, send_col = st.columns([6, 1])
with chat_col:
    user_question = st.text_input(
        "chat_input",
        placeholder="Ask anything about the code... e.g. 'What does line 5 do?' or 'Is there a faster way?'",
        label_visibility="collapsed",
    )
with send_col:
    send_btn = st.button("➤ Send")

if send_btn and user_question.strip():
    ctx_code = code_input.strip() or st.session_state.last_code
    if not ctx_code:
        st.warning("⚠️  Paste some code above first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # RAG retrieve
        st.session_state.rag.index_code(ctx_code, detected, chunk_size=256)
        rag_ctx = st.session_state.rag.retrieve(user_question, top_k=3)

        with st.spinner("Thinking..."):
            answer = st.session_state.llm.chat(
                history  = st.session_state.chat_history[:-1],
                question = user_question,
                code     = ctx_code,
                context  = rag_ctx,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()