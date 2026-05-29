# Contributing to AI Use-Cases Showcase

Thank you for your interest in contributing! This guide explains how to add new use-cases, improve existing ones, or integrate alternative AI models.

---

## 📋 Table of Contents

1. [Adding a New Use-Case](#1-adding-a-new-use-case)
2. [Folder Naming Convention](#2-folder-naming-convention)
3. [Required Files per Use-Case](#3-required-files-per-use-case)
4. [Supporting Multiple AI Backends](#4-supporting-multiple-ai-backends)
5. [Writing Beginner-Friendly Code](#5-writing-beginner-friendly-code)
6. [Adding a New Domain Folder](#6-adding-a-new-domain-folder)
7. [Submitting a Pull Request](#7-submitting-a-pull-request)

---

## 1. Adding a New Use-Case

Each use-case lives in its own sub-folder inside a domain folder (`manufacturing/` or `finance/`).

**Steps:**
```
1. Create a new folder:  <domain>/<NN>-<short-name>/
2. Add your Python script(s)
3. Add sample data under data/
4. Add a README.md explaining the use-case
5. Update the domain README.md table
6. Update the root README.md table
```

---

## 2. Folder Naming Convention

```
<domain>/<two-digit-number>-<kebab-case-name>/
```

Examples:
```
manufacturing/05-energy-optimization/
finance/05-earnings-predictor/
```

---

## 3. Required Files per Use-Case

Every use-case folder **must** contain:

```
<use-case-folder>/
├── README.md              # What it does, how to run, sample output
├── <script_name>.py       # Main Python script
└── data/
    └── sample_*.csv/.json # Small sample dataset (no real PII!)
```

### README.md template

```markdown
# <Use-Case Title>

## What This Does
<One paragraph description>

## How to Run
\```bash
python <script_name>.py
\```

## Expected Input
<Describe the columns / fields in sample data>

## Sample Output
\```
<paste a real sample output here>
\```

## How It Works
<Step-by-step explanation>

## Switching AI Providers
<Explain the AI_PROVIDER env variable>
```

---

## 4. Supporting Multiple AI Backends

All scripts must support **both** OpenAI and Ollama. Use this pattern:

```python
import os

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "openai" or "ollama"

def call_ai(prompt: str) -> str:
    if AI_PROVIDER == "openai":
        return call_openai(prompt)
    return call_ollama(prompt)
```

### Adding a new backend (e.g., Anthropic Claude)

1. Add a `call_anthropic(prompt)` function
2. Extend the `call_ai()` dispatcher
3. Document it in your README.md
4. Add the new package to `requirements.txt`

---

## 5. Writing Beginner-Friendly Code

Do:
- Add a docstring to every function explaining what it does
- Use clear, descriptive variable names
- Add inline comments for non-obvious logic
- Print progress messages (`print("Analyzing...")`)
- Keep functions short (< 30 lines each)

Avoid:
- Complex one-liners that are hard to read
- Unexplained abbreviations
- Hardcoded API keys (use environment variables!)
- Dependencies that are hard to install

---

## 6. Adding a New Domain Folder

Want to add a new industry (e.g., `healthcare/`, `logistics/`)?

1. Create the folder: `healthcare/`
2. Add a `healthcare/README.md` describing the domain and listing use-cases
3. Add at least one use-case sub-folder following the conventions above
4. Update the root `README.md` to include the new domain

---

## 7. Submitting a Pull Request

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/my-new-use-case`
3. Commit your changes with clear messages
4. Open a Pull Request with:
   - A description of the use-case
   - A sample output screenshot or text
   - Confirmation that you tested with at least one AI backend

---

Happy building!
