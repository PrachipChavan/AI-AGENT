# AI-AGENT
🚀 New AI Project: Multi-Agent Plagiarism-Free Content Rewriter ✍️🤖

Excited to share my latest project built using Groq LLM + Streamlit + Multi-Agent Architecture!

I created a smart rewriting tool that transforms any text into a fully plagiarism-free, meaning-preserved version. It supports uploading PDF, DOCX, TXT, controls tone & length, preserves keywords, detects similarity, and lets users download the final output in PDF or TXT.

Powered by:
🔹 Groq LLM (super-fast generation)
🔹 Python + Streamlit UI
🔹 Multi-Agent System (Extractor, Rewriter, Polisher)
🔹 ReportLab PDF Generator
🚀 Key Features
🔹 1. Multi-Agent Rewrite System (CrewAI-based, optional)

Extractor Agent → Identifies key facts

Rewriter Agent → Rewrites with chosen tone & length

Polisher Agent → Improves clarity, structure, and fluency

🔹 2. Groq LLM (Ultra-Fast Rewriting)

Uses llama-3.1-8b-instant for real-time generation

Ensures high-quality output in seconds

🔹 3. Multiple Input Sources

Upload PDF, DOCX, or TXT

OR paste text manually

Automatic extraction from uploaded files

🔹 4. Custom Rewriting Controls

Tone: Neutral, Formal, Casual, Persuasive, Technical

Length: Short, Medium, Long

Preserve keywords/phrases

Option to include citation placeholders

Prevents model from inventing fake data

🔹 5. Plagiarism Similarity Score

Powered by Jaccard similarity using shingle-based comparison

Shows % similarity between original & rewritten text

Indicators:

🟥 High (50%+): needs more rewriting

🟧 Medium (20–49%): consider tweaks

🟩 Low (<20%): strong originality

🔹 6. Export Options

Download rewritten text as PDF

Or as simple TXT file

PDF generated using ReportLab

🔹 7. Modern Streamlit UI

Clean card-based layout

Responsive two-column design

Attractive fonts, colors, and spacing

Developer logs included for debugging
