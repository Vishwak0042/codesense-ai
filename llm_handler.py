"""
llm_handler.py — LLM Integration Layer
Groq API is hardcoded — no key needed from user.
"""

from typing import List, Dict

# ─────────────────────────────────────────────
# HARDCODE YOUR GROQ API KEY HERE
# Get a free key at: https://console.groq.com
# ─────────────────────────────────────────────
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are CodeSense AI, an expert software engineer and CS educator.
Help developers understand, summarize, debug, and optimize code.

Guidelines:
- Use clear, plain English.
- Structure responses with headers and bullet points using markdown.
- Be thorough but concise.
- Always be educational and beginner-friendly.
"""

ACTION_PROMPTS = {
    "Explanation": """Explain the following {language} code in detail.
Structure your answer as:
1. **Overview** - What does this code do?
2. **Key Components** - Main functions, classes, variables.
3. **Line-by-Line Walkthrough** - Explain each important part.
4. **How It Works** - The overall flow and logic.
5. **Example Use Case** - When would you use this?

CODE:
```{language}
{code}
```
{context_section}""",

    "Summarization": """Summarize the following {language} code concisely.
Cover:
- **Purpose** — What it does in 1-2 sentences.
- **Inputs & Outputs** — What goes in and comes out.
- **Core Logic** — The main algorithm or approach.
- **Libraries Used** — Any imports or dependencies.
- **Complexity** — Time/space if relevant.

CODE:
```{language}
{code}
```
{context_section}""",

    "Debugging": """Analyze the following {language} code for bugs and issues.
Provide:
1. **Issues Found** — List every bug or problem.
2. **Root Cause** — Why each issue happens.
3. **Fixed Code** — Show the corrected version.
4. **Edge Cases** — What inputs might break it?
5. **Improvements** — Any style or safety improvements.

If no bugs found, say so and suggest defensive improvements.

CODE:
```{language}
{code}
```
{context_section}""",

    "Optimization": """Optimize the following {language} code for performance and readability.
Cover:
1. **Bottlenecks** — What is slow or inefficient?
2. **Optimized Code** — Rewritten improved version.
3. **Complexity Comparison** — Before vs after (time/space).
4. **Readability Tips** — Better naming, structure, style.
5. **Modern Alternatives** — Idiomatic language features to use.

CODE:
```{language}
{code}
```
{context_section}""",
}


class LLMHandler:
    """Calls Groq API with a hardcoded key. No user input needed."""

    def __init__(self):
        import openai
        self._client = openai.OpenAI(
            api_key  = GROQ_API_KEY,
            base_url = GROQ_BASE_URL,
        )

    def _call(self, messages: List[Dict], max_tokens: int = 1500) -> str:
        response = self._client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            temperature = 0.3,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content.strip()

    def run(self, code: str, language: str, action: str, context: str = "") -> str:
        """Run Explain / Summarize / Debug / Optimize."""
        context_section = (
            f"\n**Relevant Context:**\n{context}\n" if context.strip() else ""
        )
        prompt = ACTION_PROMPTS[action].format(
            language        = language,
            code            = code,
            context_section = context_section,
        )
        try:
            return self._call([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ])
        except Exception as e:
            return f"⚠️ Error: {e}"

    def chat(self, history: List[Dict], question: str, code: str, context: str = "") -> str:
        """Multi-turn Q&A."""
        context_block = f"\nContext:\n{context}\n" if context.strip() else ""
        system = (
            SYSTEM_PROMPT
            + f"\n\nThe user is asking about this code:\n```\n{code[:2000]}\n```"
            + context_block
        )
        messages = [{"role": "system", "content": system}]
        for msg in history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})
        try:
            return self._call(messages)
        except Exception as e:
            return f"⚠️ Error: {e}"
