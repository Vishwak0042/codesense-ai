"""
utils.py — Shared Utility Functions
=====================================
Helper functions used across the application:
  - Language auto-detection
  - Code formatting/display
  - Chat history export
  - Token estimation
"""

import re
import json
from typing import List, Dict


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

# Heuristic patterns per language
_LANG_PATTERNS: Dict[str, List[str]] = {
    "Python": [
        r"\bdef\s+\w+\s*\(",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r":\s*$",
        r"\bprint\s*\(",
        r"\bself\b",
        r"\bif\s+__name__\s*==",
        r"#.*$",                    # Python comments
    ],
    "JavaScript": [
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bfunction\s+\w+\s*\(",
        r"\bconsole\.log\s*\(",
        r"=>\s*{",
        r"\bmodule\.exports\b",
        r"\brequire\s*\(",
        r"\/\/.*$",                 # JS line comment
    ],
    "TypeScript": [
        r":\s*string\b",
        r":\s*number\b",
        r":\s*boolean\b",
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
        r"<\w+>",                   # generics
    ],
    "Java": [
        r"\bpublic\s+(class|static|void)\b",
        r"\bSystem\.out\.print",
        r"\bprivate\s+\w+\s+\w+",
        r"\bnew\s+\w+\s*\(",
        r"@Override\b",
        r"\bimport\s+java\.",
    ],
    "C++": [
        r"#include\s*<",
        r"\bstd::",
        r"\bcout\s*<<",
        r"\bcin\s*>>",
        r"::\w+",
        r"\bnamespace\s+\w+",
        r"\btemplate\s*<",
    ],
    "Go": [
        r"\bfunc\s+\w+\s*\(",
        r"\bpackage\s+\w+",
        r":=",
        r"\bfmt\.Print",
        r"\bgo\s+\w+\(",
    ],
    "Ruby": [
        r"\bdef\s+\w+",
        r"\bend\b",
        r"\bputs\b",
        r"\battr_accessor\b",
        r"\brequire\s+'",
        r"do\s*\|",
    ],
    "Rust": [
        r"\bfn\s+\w+\s*\(",
        r"\blet\s+mut\b",
        r"\bprintln!\s*\(",
        r"\buse\s+std::",
        r"\bimpl\s+\w+",
        r"->\s*\w+",
    ],
}


def detect_language(code: str) -> str:
    """
    Heuristically detect the programming language of a code snippet.

    Parameters
    ----------
    code : Source code string.

    Returns
    -------
    Name of the detected language, or 'Unknown'.
    """
    scores: Dict[str, int] = {lang: 0 for lang in _LANG_PATTERNS}

    for lang, patterns in _LANG_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            scores[lang] += len(matches)

    # TypeScript must score higher than JavaScript to win
    if scores.get("TypeScript", 0) > 2 and scores.get("TypeScript", 0) >= scores.get("JavaScript", 0):
        return "TypeScript"

    best_lang  = max(scores, key=lambda k: scores[k])
    best_score = scores[best_lang]

    return best_lang if best_score > 0 else "Unknown"


# ─────────────────────────────────────────────
# CODE DISPLAY FORMATTING
# ─────────────────────────────────────────────

def format_code_display(code: str, max_lines: int = 50) -> str:
    """
    Truncate very long code snippets for display purposes.

    Parameters
    ----------
    code      : Raw source code.
    max_lines : Maximum lines to show before truncating.

    Returns
    -------
    Possibly-truncated code string with a notice appended.
    """
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    truncated = "\n".join(lines[:max_lines])
    return truncated + f"\n\n... [{len(lines) - max_lines} more lines truncated] ..."


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count (1 token ≈ 4 characters for English/code).

    Parameters
    ----------
    text : Any string.

    Returns
    -------
    Estimated token count as integer.
    """
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int = 3000) -> str:
    """
    Truncate text to approximately max_tokens tokens.

    Parameters
    ----------
    text       : Source text.
    max_tokens : Token budget.

    Returns
    -------
    Truncated text string.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for token limit] ..."


# ─────────────────────────────────────────────
# CHAT HISTORY UTILITIES
# ─────────────────────────────────────────────

def save_chat_history(history: List[Dict], filepath: str = "chat_history.json") -> bool:
    """
    Save chat history to a JSON file.

    Parameters
    ----------
    history  : List of {role, content} message dicts.
    filepath : Output file path.

    Returns
    -------
    True on success, False on failure.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_chat_history(filepath: str = "chat_history.json") -> List[Dict]:
    """
    Load chat history from a JSON file.

    Parameters
    ----------
    filepath : Path to the JSON file.

    Returns
    -------
    List of message dicts, or empty list on failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def format_history_for_display(history: List[Dict]) -> str:
    """
    Format chat history into a readable plain-text transcript.

    Parameters
    ----------
    history : List of {role, content} dicts.

    Returns
    -------
    Formatted string transcript.
    """
    lines = []
    for msg in history:
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        lines.append(f"[{role}]\n{content}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MISC HELPERS
# ─────────────────────────────────────────────

def clean_code(code: str) -> str:
    """
    Strip leading/trailing whitespace and normalize line endings.

    Parameters
    ----------
    code : Raw code string.

    Returns
    -------
    Cleaned code string.
    """
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code.strip()


def extract_code_blocks(markdown_text: str) -> List[str]:
    """
    Extract fenced code blocks from markdown text.

    Parameters
    ----------
    markdown_text : Markdown string potentially containing ``` blocks.

    Returns
    -------
    List of code block strings (without fences).
    """
    pattern = r"```(?:\w+)?\n?(.*?)```"
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]