# backend/prompts.py

SUMMARY_PROMPT = """
You are an assistant that summarizes annual reports. Produce a concise executive summary covering:
- Company overview & business segments
- Financial highlights (revenue, profit, margins) — mention absolute numbers if present
- Key strategic initiatives and risks
- Outlook / guidance
- One-line “Key takeaway” for executives

Be precise, use bullet points and keep it under 400 words.
"""

QA_PROMPT = """
You are an assistant answering questions using ONLY the context below from the annual report.
If the answer is not present in the context, say "I don't see that in the provided report."
Context:
{context}

Question: {question}

Answer concisely and cite which snippet(s) you used by including the first 30 characters of the snippet in your answer for traceability.
"""
