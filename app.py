# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
import time
import re
import traceback
from xml.sax.saxutils import escape
from typing import Optional

import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
import docx2txt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib import colors

# -------------------------
# Config / Owner
# -------------------------
OWNER_NAME = os.getenv("OWNER_NAME", "Prachi Chavan")
USE_OFFICIAL_CREW_ENV = os.getenv("USE_OFFICIAL_CREW", "false").lower() in ("1", "true", "yes")
USE_CREWAI = False
if USE_OFFICIAL_CREW_ENV:
    try:
        # optional import; app will still run without CrewAI installed
        from crewai import Agent, Crew, Process
        from crewai.project import CrewBase, agent, crew, llm
        USE_CREWAI = True
    except Exception:
        USE_CREWAI = False

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="✍️ Multi Agent Plagiarism-Free Content Rewriter",
    page_icon="✍️",
    layout="wide"
)

# -------------------------
# Groq client
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.info("GROQ_API_KEY not set — Groq LLM features will be disabled or fallback.")
client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# -------------------------
# Session defaults
# -------------------------
defaults = {
    "source_text": "",
    "uploaded_file_name": "",
    "rewritten": "",
    "tone": "Neutral",
    "length": "Medium",
    "preserve_keywords": "",
    "include_citations": False,
    "plagiarism_score": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -------------------------
# Styles - attractive UI
# -------------------------
st.markdown(
    """
    <style>
    :root { --accent: #0b3d91; --bg: #f7fbff; --card: #ffffff; }
    body { background: var(--bg); font-family: Inter, Arial, Helvetica, sans-serif; }
    .header { font-size:30px; font-weight:900; color:var(--accent); margin-bottom:6px; }
    .sub { color:#4b5563; margin-bottom:14px; }
    .owner { font-weight:700; color:var(--accent); font-size:15px; text-align:right; }
    .card { background:var(--card); border-radius:14px; padding:18px; border:1px solid #e6eef8; box-shadow: 0 6px 18px rgba(11,61,145,0.06); }
    .muted { color:#6b7280; font-size:13px; }
    .small { font-size:13px; color:#6b7280; }
    .btn { background:#0b3d91; color:white; padding:8px 16px; border-radius:10px; font-weight:700; }
    .result-box { max-height:480px; overflow-y:auto; padding:14px; border-radius:10px; border:1px solid #e6eef8; background:#ffffff; color:#111827; }
    .inline-label { display:inline-block; min-width:110px; font-weight:600; color: #0b3d91; }
    /* Horizontal name-like buttons placeholder style (if needed) */
    .hbutton { display:inline-block; margin-right:8px; margin-bottom:8px; padding:8px 12px; background:#f1f8ff; border-radius:10px; border:1px solid #d9e8fb; color:#0b3d91; font-weight:600; cursor:pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header row (title + owner)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div class='header'>✍️Multi Agent Plagiarism-Free Content Rewriter</div>
        <div class='sub'>Rewrite content to be plagiarism-free while keeping meaning — powered by Groq + CrewAI</div>
      </div>
      <div class='owner'>Owner: {escape(OWNER_NAME)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers: Groq chat wrapper
# -------------------------
def groq_chat(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.5, max_retries: int = 2) -> str:
    """Call Groq (if configured). Returns text or empty string on failure."""
    if not client:
        return ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            choice = resp.choices[0]
            message = getattr(choice, "message", None)
            if message:
                try:
                    return message.content
                except Exception:
                    try:
                        return message["content"]
                    except Exception:
                        pass
            for attr in ("text", "content"):
                if hasattr(choice, attr):
                    return getattr(choice, attr)
            return str(resp)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.6 + attempt * 0.5)
                continue
            # on final failure return empty and log
            st.error("Groq request failed: " + str(e))
            st.error(traceback.format_exc())
            return ""

# -------------------------
# Helpers: PDF builder
# -------------------------
def build_pdf_bytes(title: str, content: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=25*mm, bottomMargin=20*mm)
    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#0b3d91"))
    normal = ParagraphStyle("normal", fontName="Helvetica", fontSize=11, leading=14)
    story = [Paragraph(escape(title), title_style), Spacer(1, 8)]
    for block in content.split("\n\n"):
        story.append(Paragraph(escape(block).replace("\n", "<br/>"), normal))
        story.append(Spacer(1, 8))
    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------
# Helpers: file load
# -------------------------
def load_text_from_file(uploaded) -> str:
    name = uploaded.name.lower()
    uploaded.seek(0)
    if name.endswith(".txt"):
        try:
            b = uploaded.read()
            if isinstance(b, bytes):
                return b.decode("utf-8", errors="ignore")
            return str(b)
        except Exception:
            return ""
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded)
            txt = ""
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    txt += page_text + "\n"
            return txt
        except Exception:
            return ""
    if name.endswith(".docx"):
        try:
            return docx2txt.process(uploaded)
        except Exception:
            return ""
    return ""

# -------------------------
# Helpers: plagiarism estimator (shingle Jaccard)
# -------------------------
def text_shingles(s: str, k: int = 5) -> set:
    tokens = re.findall(r"\w+", s.lower())
    if len(tokens) < k:
        return set([" ".join(tokens)]) if tokens else set()
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingles.add(" ".join(tokens[i : i + k]))
    return shingles

def plagiarism_estimate(original: str, rewritten: str) -> float:
    if not original.strip() or not rewritten.strip():
        return 0.0
    a = text_shingles(original, k=5)
    b = text_shingles(rewritten, k=5)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b) if (a | b) else 1
    j = inter / uni
    return round(j * 100.0, 1)

# -------------------------
# CrewAI multi-agent (optional)
# -------------------------
if USE_CREWAI:
    try:
        @CrewBase
        class RewriterCrew:
            @llm
            def groq_llm(self):
                return {"provider": "groq"}

            @agent
            def extractor(self):
                return Agent(role="Extractor", goal="Extract key facts/keywords from the source text")

            @agent
            def rewriter(self):
                return Agent(role="Rewriter", goal="Rewrite text keeping facts, tone and length")

            @agent
            def polisher(self):
                return Agent(role="Polisher", goal="Polish the rewritten text for clarity and fluency")

            @crew
            def crew(self):
                return Crew(agents=self.agents, process=Process.sequential, verbose=False)

        def run_official_crew_rewrite(source: str, tone: str, length: str, preserve_keywords: str, include_citations: bool) -> str:
            sc = RewriterCrew()
            crew_obj = sc.crew()
            inputs = {
                "source": source,
                "tone": tone,
                "length": length,
                "preserve_keywords": preserve_keywords,
                "include_citations": include_citations,
            }
            result = crew_obj.kickoff(inputs=inputs)
            # normalize common outputs
            out = ""
            for key in ("rewritten", "polisher", "rewriter", "result", "output"):
                candidate = result.get(key) if isinstance(result, dict) else None
                if candidate:
                    out = candidate
                    break
            if not out:
                out = str(result)
            return out.strip()
    except Exception:
        USE_CREWAI = False

# -------------------------
# Layout: left column = inputs, right column = outputs
# -------------------------
left, right = st.columns([0.95, 1.05])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 1) Provide Source Content")
    uploaded = st.file_uploader("Upload source file (TXT, PDF, DOCX) or skip to paste:", type=["txt", "pdf", "docx"])
    if uploaded:
        st.session_state.uploaded_file_name = uploaded.name
        try:
            text_val = load_text_from_file(uploaded)
            if text_val:
                st.session_state.source_text = text_val
        except Exception:
            st.error("Failed to read uploaded file. Try a smaller file or plain TXT.")
    st.session_state.source_text = st.text_area("Paste source text here:", value=st.session_state.source_text, height=220)

    st.markdown("---")
    st.markdown("### 2) Rewriting Options")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.session_state.tone = st.selectbox(
            "Tone",
            ["Neutral", "Formal", "Casual", "Persuasive", "Technical"],
            index=["Neutral", "Formal", "Casual", "Persuasive", "Technical"].index(st.session_state.tone)
        )
    with col_b:
        st.session_state.length = st.selectbox(
            "Length",
            ["Short", "Medium", "Long"],
            index=["Short", "Medium", "Long"].index(st.session_state.length)
        )

    st.text_input("Preserve exact keywords/phrases (comma-separated)", value=st.session_state.preserve_keywords, key="preserve_keywords")
    st.checkbox("Ask model to include citations / sources (use '[source needed]' instead of inventing)", value=st.session_state.include_citations, key="include_citations")

    st.markdown("---")
    st.markdown("### 3) Run")
    c1, c2 = st.columns([1, 1])
    with c1:
        rewrite_click = st.button("✍️ Rewrite Content", use_container_width=True)
    with c2:
        clear_click = st.button("🧹 Clear Inputs", use_container_width=True)

    if clear_click:
        for k in ("source_text", "uploaded_file_name", "rewritten", "preserve_keywords", "plagiarism_score"):
            st.session_state[k] = defaults.get(k, "")
        st.success("Inputs cleared")

    # Run rewrite logic
    if rewrite_click:
        if not st.session_state.source_text.strip():
            st.warning("Please paste or upload source text first.")
        else:
            with st.spinner("Generating plagiarism-free rewrite..."):
                try:
                    preserve = st.session_state.preserve_keywords.strip()
                    instr_lines = [
                        "You are a professional content rewriter.",
                        "Goal: Produce a rewritten version of the SOURCE text that is plagiarism-free while preserving the original meaning and facts.",
                        f"Tone: {st.session_state.tone}.",
                        f"Length: {st.session_state.length}.",
                    ]
                    if preserve:
                        instr_lines.append(f"Preserve these keywords/phrases exactly (if present): {preserve}.")
                    if st.session_state.include_citations:
                        instr_lines.append("If you reference facts that need sourcing, note '[source needed]' instead of inventing sources.")
                    instr_lines.append("Do NOT invent numerical data. Keep named entities unchanged unless asked to generalize.")
                    instruction = "\n".join(instr_lines)

                    rewritten = ""
                    # Try official CrewAI if configured
                    if USE_CREWAI:
                        try:
                            rewritten = run_official_crew_rewrite(
                                st.session_state.source_text,
                                st.session_state.tone,
                                st.session_state.length,
                                preserve,
                                st.session_state.include_citations,
                            )
                        except Exception:
                            rewritten = ""
                    # fallback to Groq (if available)
                    if not rewritten:
                        prompt = f"{instruction}\n\nSOURCE:\n{st.session_state.source_text}\n\nRewritten:"
                        rewritten = groq_chat(prompt, temperature=0.55)
                        if not rewritten:
                            # simple fallback: collapse whitespace and return truncated text so UI remains responsive
                            src = st.session_state.source_text
                            rewritten = re.sub(r"\s+", " ", src).strip()
                            rewritten = rewritten[: min(len(rewritten), 2000)]

                    st.session_state.rewritten = rewritten.strip()
                    st.session_state.plagiarism_score = plagiarism_estimate(st.session_state.source_text, st.session_state.rewritten)
                    st.success("Rewrite complete!")
                except Exception as e:
                    st.error("Rewrite failed: " + str(e))
                    st.error(traceback.format_exc())

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Output")
    if st.session_state.rewritten:
        st.markdown(
            f"<div class='small'><span class='inline-label'>Tone:</span> {escape(st.session_state.tone)} &nbsp;&nbsp; "
            f"<span class='inline-label'>Length:</span> {escape(st.session_state.length)}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br/>", unsafe_allow_html=True)
        # result box with readable text color
        st.markdown(f"<div class='result-box'>{st.session_state.rewritten.replace(chr(10), '<br><br>')}</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        score = st.session_state.plagiarism_score or 0.0
        st.markdown(f"**Estimated similarity:** {score}% (lower = less similar to source)")
        if score >= 50:
            st.warning("High similarity detected — consider running with a stronger rewrite (different tone/length) or editing manually.")
        elif score >= 20:
            st.info("Some similarity detected — you may want to tweak phrasing for safety.")
        else:
            st.success("Low similarity — likely well rewritten (automatic estimator).")
        st.markdown("</div>", unsafe_allow_html=True)

        # Download buttons
        pdf_data = build_pdf_bytes("Rewritten Content", st.session_state.rewritten)
        st.download_button("📄 Download Rewritten (PDF)", data=pdf_data, file_name="rewritten_content.pdf", mime="application/pdf", use_container_width=True)
        st.download_button("📋 Download Rewritten (TXT)", data=st.session_state.rewritten, file_name="rewritten.txt", mime="text/plain", use_container_width=True)
    else:
        st.markdown("<div class='muted'>No rewritten content yet — generate by pressing <strong>✍️ Rewrite Content</strong> on the left.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer / debug log
# -------------------------
st.markdown("---")
with st.expander("Developer info / logs (expand)"):
    st.write({
        "USE_CREWAI": USE_CREWAI,
        "GROQ_CONFIGURED": bool(client),
        "OWNER_NAME": OWNER_NAME,
    })
    if st.button("Show last exception trace (if any)", key="trace"):
        st.text(traceback.format_exc())
