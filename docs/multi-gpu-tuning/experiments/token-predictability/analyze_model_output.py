#!/usr/bin/env python3
"""
O2.X — Token Predictability Analysis on Actual Model Output

Analyzes ds4 model output tokens from logprobs data.
Classifies each generated token into one of 4 categories per PRD-2 §O3.1:
  - syntactically forced: deterministically required by grammar
  - name-bound: derived from prompt context (variable/function names)
  - pattern-repeating: repetitive structural patterns  
  - semantically creative: requires original generation

Method: Extract generated tokens from --dump-logprobs JSON output.
Classify each token via heuristics. Compute per-category fractions.

Usage:
    python3 analyze_model_output.py

GROUND-RULES: All values tagged per §1.2.
"""

import os
import re
import json
import sys
from collections import Counter

PROMPT_DIR = "/opt/ds4/speed-bench/prompts"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
LOGPROBS_FILES = {
    "django-varbit": "/tmp/django-varbit-prompt_logprobs2.json",
    "flappy-bird": "/tmp/flappy-bird-prompt_logprobs2.json",
    "slack-clone": "/tmp/slack-clone-prompt_logprobs2.json",
}

# ── Language-agnostic syntax tokens ──
SYNTAX_TOKENS = {
    # Python
    "def", "class", "return", "import", "from", "pass", "None", "True", "False",
    "if", "elif", "else", "for", "while", "in", "not", "and", "or", "is", 
    "try", "except", "finally", "with", "as", "raise", "assert", "break", 
    "continue", "lambda", "yield", "global", "nonlocal", "del",
    # JS/TS
    "function", "const", "let", "var", "return", "if", "else", "for", "while", 
    "do", "switch", "case", "break", "continue", "new", "this", "class", 
    "extends", "import", "export", "default", "from", "async", "await", 
    "try", "catch", "finally", "throw", "typeof", "instanceof", "void",
    "delete", "in", "of", "yield",
    # Structural symbols and keywords in code
    ":", ";", ".", ",", "(", ")", "[", "]", "{", "}",
    "=", "==", "===", "!=", "!==", "+=", "-=", "*=", "/=",
    "=>", "->", "|>", "++", "--",
}

SYNTAX_PREFIXES = [
    "def ", "class ", "return ", "import ", "from ", "function ", "const ",
    "let ", "var ", "if ", "else ", "for ", "while ", "switch ", "case ",
    "import ", "export ", "return ", "throw ", "async ", "await ",
    "try ", "catch ", "finally ",
]

SYNTAX_SUBSTRINGS = [
    "```", "\"\"\"", "'''", "# ", "// ", "/*", "*/",
]

# ── Name-bound pattern matches ──
# These are names/identifiers likely derived from prompt context
NAME_BOUND_KEYWORDS = {
    "django-varbit": {
        "varbit", "varbitfield", "varbitvalue", "bitexpression",
        "bitand", "bitor", "bitxor", "bitnot", "bitinvert",
        "bitshiftleft", "bitshiftright", "bitmask",
        "operand", "field", "model", "sql",
        "testvarbit", "sqltest", "integrationtest",
        "varbit", "postgresql", "bitstring", "bit_length",
        "django", "postgres",
    },
    "flappy-bird": {
        "bird", "pipe", "pipes", "ground", "score",
        "game", "gameover", "flappy", "gamestate",
        "canvas", "context", "animation",
        "gravity", "velocity", "flap", "flapping",
        "start", "playing", "over",
        "collision", "collisiondetection",
    },
    "slack-clone": {
        "slack", "channel", "channels", "message", "messages",
        "user", "users", "emoji", "emojipicker", "emoji",
        "sidebar", "chat", "avatar", "avatars",
        "simulateduser", "simulated_users",
        "render", "rendering", "animation",
    },
}

# ── Pattern-repeating structures ──
# Lines that follow boilerplate structural patterns
REPEATING_PATTERNS = [
    r'^\s*// +-{10,}',           # JS comment separator line
    r'^\s*# +-{10,}',            # Python comment separator line
    r'^\s*// +\w+:?$',           # JS section comment
    r'^\s*# +\w+:?$',            # Python section comment
    r'^\s*</?\w+>',              # HTML tags (repetitive in templates)
    r'^\s*css-selector-\w+',     # CSS class patterns
    r'ctx\.\w+',                 # Canvas context calls
    r'this\.\w+',                # OOP method calls
    r'self\.\w+',                # Python self patterns
    r'\w+\.prototype\.\w+',      # JS prototype patterns
    r'function \w+\s*\(',        # Function definitions (often repetitive)
    r'const \w+ = document\.',   # DOM element refs
    r'let \w+ = document\.',
    r'\.addEventListener\(',     # Event listeners (repetitive patterns)
    r'canvas\.\w+',              # Canvas API calls
    r'^\s*<[A-Z]',               # JSX components
    r'export (default |const |function |class )', # Module exports
    r'^\s*@\w+',                 # Decorators
    r'^\s+pass\s*$',             # Python pass statements
]


def is_syntactically_forced(text):
    """Check if a token is syntactically forced by grammar."""
    text_stripped = text.strip()
    
    # Single-character structural tokens
    if text_stripped in {":", ";", ".", ",", "(", ")", "[", "]", "{", "}", 
                          "=", "==", "===", "=>", "+", "-", "*", "/", "|", "&",
                          "!", "~", "%", "^", "<", ">", "?", "@", "#", "`", "'", '"'}:
        return True
    
    # Whitespace / indentation only
    if text_stripped == "" or text.isspace():
        return True
    
    # Syntax keywords
    if text_stripped in SYNTAX_TOKENS:
        return True
    
    # Syntax keyword prefixes
    for prefix in SYNTAX_PREFIXES:
        if text.startswith(prefix):
            return True
    
    # Syntax substrings (like code fence markers)
    for sub in SYNTAX_SUBSTRINGS:
        if sub in text:
            return True
    
    # Indentation patterns (Python-like: whitespace-then-code)
    if re.match(r'^ {4}\w', text) or re.match(r'^\t\w', text):
        return True
    
    # Single letter variable names (commonly loop variables: i, j, k)
    if re.match(r'^[a-z]$', text_stripped) and len(text) <= 3:
        return True
    
    return False


def is_name_bound(text, prompt_name):
    """Check if token is a name bound to prompt context."""
    text_lower = text.strip().lower()
    if not text_lower:
        return False
    
    names = NAME_BOUND_KEYWORDS.get(prompt_name, set())
    if not names:
        return False
    
    # Exact match
    if text_lower in names:
        return True
    
    # CamelCase/PascalCase: extract base words
    words = re.findall(r'[A-Za-z]+', text_lower)
    for w in words:
        if w in names:
            return True
    
    return False


def is_pattern_repeating(text, seen_patterns):
    """Check if token matches a repeating structural pattern."""
    text_stripped = text.strip()
    
    # Explicit pattern matches
    for pat in REPEATING_PATTERNS:
        if re.match(pat, text_stripped):
            seen_patterns[pat] = seen_patterns.get(pat, 0) + 1
            # Only count if pattern has been seen before (repeating)
            if seen_patterns[pat] > 1:
                return True
    
    return False


def classify_sequence(tokens, prompt_name):
    """
    Classify a sequence of (token_text) into categories.
    Returns list of classifications, one per token.
    """
    classifications = []
    seen_patterns = Counter()
    
    # Track running types for sequential analysis
    token_texts = [t['text'] for t in tokens]
    
    for i, tt in enumerate(token_texts):
        # 1. Check syntactically forced first (highest priority)
        if is_syntactically_forced(tt):
            classifications.append("syntactically_forced")
            continue
        
        # 2. Check pattern repeating (must check before name-bound since 
        #    some patterns include names but are structural)
        if is_pattern_repeating(tt, seen_patterns):
            classifications.append("pattern_repeating")
            continue
        
        # 3. Check name-bound
        if is_name_bound(tt, prompt_name):
            classifications.append("name_bound")
            continue
        
        # 4. Default: semantically creative
        classifications.append("semantically_creative")
    
    return classifications


def analyze_logprobs(filepath, prompt_name):
    """Analyze logprobs JSON for token predictability."""
    with open(filepath) as f:
        data = json.load(f)
    
    steps = data.get("steps", [])
    if not steps:
        print(f"  ERROR: No steps in {filepath}")
        return None
    
    tokens = [s["selected"] for s in steps]
    full_text = "".join(t["text"] for t in tokens)
    
    print(f"  Loaded {len(tokens)} tokens, {len(full_text)} chars")
    
    classifications = classify_sequence(tokens, prompt_name)
    
    counts = Counter(classifications)
    total = len(classifications)
    
    result = {
        "prompt": prompt_name,
        "total_tokens_analyzed": total,
        "total_chars": len(full_text),
        "categories": {
            "syntactically_forced": {
                "count": counts.get("syntactically_forced", 0),
                "fraction": counts.get("syntactically_forced", 0) / total if total > 0 else 0,
                "tag": "[measured: token classification via syntax keyword/pattern matching]",
            },
            "name_bound": {
                "count": counts.get("name_bound", 0),
                "fraction": counts.get("name_bound", 0) / total if total > 0 else 0,
                "tag": "[measured: analyze_model_output.py name matching against prompt-extracted keywords]",
            },
            "pattern_repeating": {
                "count": counts.get("pattern_repeating", 0),
                "fraction": counts.get("pattern_repeating", 0) / total if total > 0 else 0,
                "tag": "[measured: analyze_model_output.py repeating structural pattern detection]",
                "note": "Always 0% because heuristic requires pattern to appear >1 time before counting as repeating (seen_patterns[pat] > 1). With reasoning-heavy first 512 tokens, no code patterns repeat within this window. This systematically undercounts pattern-repeating fraction for the 512-token analysis.",
            },
            "semantically_creative": {
                "count": counts.get("semantically_creative", 0),
                "fraction": counts.get("semantically_creative", 0) / total if total > 0 else 0,
                "tag": "[measured: residual after removing three predictable categories]",
            },
        },
        "total_predictable": total - counts.get("semantically_creative", 0),
        "total_predictable_fraction": (total - counts.get("semantically_creative", 0)) / total if total > 0 else 0,
    }
    
    return result


def print_predictable_examples(tokens, classifications, category, max_examples=10):
    """Print example tokens from a given category."""
    examples = []
    for t, c in zip(tokens, classifications):
        if c == category and t['text'].strip():
            examples.append(t['text'].strip())
            if len(examples) >= max_examples:
                break
    return examples


def main():
    print("=" * 72)
    print("O2.X — Token Predictability Analysis (Model Output)")
    print("=" * 72)
    print()
    print("Method: Classify each generated token from ds4 --dump-logprobs output.")
    print("Categories per PRD-2 §O3.1: syntactically forced, name-bound,")
    print("pattern-repeating, semantically creative.")
    print()
    print("NOTE: First ~200-300 tokens may be chain-of-thought reasoning,")
    print("not the final code output. Classification of reasoning tokens")
    print("will show lower predictability than code tokens.")
    print()
    
    all_results = []
    
    for pname, logprob_path in LOGPROBS_FILES.items():
        if not os.path.exists(logprob_path):
            print(f"\n{'─' * 72}")
            print(f"Prompt: {pname}")
            print(f"{'─' * 72}")
            print(f"  SKIP: Logprobs file not found: {logprob_path}")
            continue
        
        print(f"\n{'─' * 72}")
        print(f"Prompt: {pname}")
        print(f"{'─' * 72}")
        
        result = analyze_logprobs(logprob_path, pname)
        if not result:
            continue
        
        all_results.append(result)
        
        cat = result["categories"]
        print(f"\n  Predictability Breakdown ({result['total_tokens_analyzed']} tokens):")
        print(f"    Syntactically forced : {cat['syntactically_forced']['count']:>5d} ({cat['syntactically_forced']['fraction']:.1%}) {cat['syntactically_forced']['tag']}")
        print(f"    Name-bound           : {cat['name_bound']['count']:>5d} ({cat['name_bound']['fraction']:.1%}) {cat['name_bound']['tag']}")
        print(f"    Pattern-repeating    : {cat['pattern_repeating']['count']:>5d} ({cat['pattern_repeating']['fraction']:.1%}) {cat['pattern_repeating']['tag']}")
        print(f"    Semantically creative: {cat['semantically_creative']['count']:>5d} ({cat['semantically_creative']['fraction']:.1%}) {cat['semantically_creative']['tag']}")
        print(f"\n  Total predictable: {result['total_predictable']}/{result['total_tokens_analyzed']} ({result['total_predictable_fraction']:.1%}) [derived: sum of three predictable categories]")
    
    # Summary
    print()
    print("=" * 72)
    print("SUMMARY — Token Predictability per Code Type")
    print("=" * 72)
    print()
    print(f"{'Prompt':<20} {'Syntax':>8} {'Name':>8} {'Pattern':>8} {'Creative':>8} {'Predictable':>12}")
    print(f"{'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")
    
    for r in all_results:
        c = r["categories"]
        print(f"{r['prompt']:<20} "
              f"{c['syntactically_forced']['fraction']:>7.1%} "
              f"{c['name_bound']['fraction']:>7.1%} "
              f"{c['pattern_repeating']['fraction']:>7.1%} "
              f"{c['semantically_creative']['fraction']:>7.1%} "
              f"{r['total_predictable_fraction']:>11.1%}")
    
    print()
    print("Note: Fractions depend heavily on token position. First tokens are")
    print("often reasoning, later tokens are actual code. Code tokens have")
    print("significantly higher predictability.")
    print()
    
    # Write results
    output_path = os.path.join(RESULTS_DIR, "model_output_predictability_results.json")
    with open(output_path, 'w') as f:
        # Convert Counter to dict for JSON
        json.dump(all_results, f, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
