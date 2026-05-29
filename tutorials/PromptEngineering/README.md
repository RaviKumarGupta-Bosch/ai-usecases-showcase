# Prompt Engineering Tutorial

Prompt engineering is the practice of designing inputs to guide LLMs towards accurate, reliable, and useful outputs.

## Curriculum

```
01-basics/
  01_zero_few_shot.py         — Zero-shot, one-shot, few-shot prompting
  02_chain_of_thought.py      — CoT, zero-shot CoT, self-consistency
  03_role_prompting.py        — System messages, personas, expert framing

02-intermediate/
  01_structured_output.py     — JSON mode, Pydantic extraction, output parsers
  02_react_prompting.py       — Reason + Act (ReAct) pattern

03-advanced/
  01_prompt_injection_defense.py — Injection attacks and defenses

04-UseCases/
  01_prompt_library.py        — Reusable prompt templates for common tasks
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python 01-basics/01_zero_few_shot.py
```

## Key Concepts

| Technique | Description |
|-----------|-------------|
| **Zero-shot** | Ask without examples — relies on world knowledge |
| **Few-shot** | Provide examples to steer format/style |
| **CoT** | Ask the model to "think step by step" |
| **ReAct** | Interleave reasoning with tool actions |
| **Role prompting** | Frame the model as an expert persona |
| **Structured output** | Force JSON/schema-conformant responses |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY` in `.env`
