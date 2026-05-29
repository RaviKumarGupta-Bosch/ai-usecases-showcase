"""
Python 01-Basics — Data Types & Structures
============================================
Topics covered:
  1. Built-in types: int, float, str, bool, None
  2. Lists — mutation, slicing, sorting
  3. Dictionaries — CRUD, merging, iteration patterns
  4. Sets — deduplication, intersection, union
  5. Tuples — immutability and unpacking
  6. List / dict / set comprehensions
  7. Practical patterns used in AI pipelines

Run:
  python 01_data_types_and_structures.py
"""

from dotenv import load_dotenv
load_dotenv()


# ── 1. Built-in scalar types ───────────────────────────────────────────────────
def demo_scalar_types():
    print("\n=== 1. Built-in Scalar Types ===")

    # Integers & floats
    tokens = 1024
    temperature = 0.7
    print(f"  tokens={tokens!r}     type={type(tokens).__name__}")
    print(f"  temperature={temperature!r}  type={type(temperature).__name__}")

    # Strings — immutable, rich methods
    model = "  gpt-4o-mini  "
    print(f"  strip:  {model.strip()!r}")
    print(f"  upper:  {model.strip().upper()!r}")
    print(f"  starts: {model.strip().startswith('gpt')}")

    # f-strings (preferred over .format())
    prompt_tokens, completion_tokens = 512, 256
    msg = f"Usage: {prompt_tokens} prompt + {completion_tokens} completion = {prompt_tokens + completion_tokens} total"
    print(f"  {msg}")

    # None & bool
    api_key = None
    is_ready = api_key is not None
    print(f"  api_key is None: {api_key is None} | is_ready: {is_ready}")


# ── 2. Lists ───────────────────────────────────────────────────────────────────
def demo_lists():
    print("\n=== 2. Lists ===")

    models = ["gpt-4o", "claude-3-opus", "llama3.2", "mistral"]

    # Indexing & slicing
    print(f"  First:   {models[0]}")
    print(f"  Last:    {models[-1]}")
    print(f"  Slice:   {models[1:3]}")
    print(f"  Reverse: {models[::-1]}")

    # Mutation
    models.append("gemini-pro")
    models.insert(0, "o1-preview")
    models.remove("mistral")
    print(f"  After mutations: {models}")

    # Sorting
    sorted_models = sorted(models, key=str.lower)
    print(f"  Sorted: {sorted_models}")

    # Unpacking
    first, *rest = models
    print(f"  first={first!r}, rest={rest}")

    # Useful list methods for AI
    scores = [0.92, 0.87, 0.95, 0.78, 0.95]
    print(f"  max={max(scores)}  min={min(scores)}  avg={sum(scores)/len(scores):.2f}")
    print(f"  unique sorted: {sorted(set(scores), reverse=True)}")


# ── 3. Dictionaries ───────────────────────────────────────────────────────────
def demo_dicts():
    print("\n=== 3. Dictionaries ===")

    # Creation & access
    config = {
        "model":       "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens":  1024,
        "stream":      True,
    }
    print(f"  model:       {config['model']}")
    print(f"  get missing: {config.get('top_p', 1.0)!r}")   # safe default

    # Add / update / delete
    config["top_p"] = 0.95
    config["temperature"] = 0.2
    del config["stream"]
    print(f"  after edits: {config}")

    # Iteration patterns
    print("  keys/values:")
    for key, value in config.items():
        print(f"    {key:<15} = {value}")

    # Merging (Python 3.9+)
    overrides = {"temperature": 0.8, "max_tokens": 512}
    merged = config | overrides
    print(f"  merged: {merged}")

    # Nested dict — common in LLM message formats
    message = {"role": "user", "content": {"type": "text", "text": "Hello!"}}
    print(f"  nested: {message['content']['text']!r}")

    # Dict comprehension — remap model names
    short_names = {k: v for k, v in {"gpt-4o-mini": "mini", "gpt-4o": "full"}.items()}
    print(f"  dict comp: {short_names}")


# ── 4. Sets ────────────────────────────────────────────────────────────────────
def demo_sets():
    print("\n=== 4. Sets ===")

    required_tools = {"search", "calculator", "calendar", "code_exec"}
    enabled_tools  = {"search", "calculator", "weather", "code_exec"}

    print(f"  available (intersection): {required_tools & enabled_tools}")
    print(f"  all tools (union):        {required_tools | enabled_tools}")
    print(f"  missing tools (diff):     {required_tools - enabled_tools}")
    print(f"  only in enabled:          {enabled_tools - required_tools}")

    # Fast membership testing — O(1) vs list O(n)
    banned_models = {"gpt-3.5-turbo", "text-davinci-003"}
    model = "gpt-4o-mini"
    print(f"  '{model}' banned: {model in banned_models}")

    # Deduplication
    tags = ["python", "ai", "python", "llm", "ai", "tutorial"]
    unique_tags = sorted(set(tags))
    print(f"  deduplicated tags: {unique_tags}")


# ── 5. Tuples & unpacking ─────────────────────────────────────────────────────
def demo_tuples():
    print("\n=== 5. Tuples & Unpacking ===")

    # Immutable records — good for fixed data
    token_usage = (512, 256, 768)   # (prompt, completion, total)
    prompt_t, completion_t, total_t = token_usage
    print(f"  prompt={prompt_t}, completion={completion_t}, total={total_t}")

    # Named return values via tuple
    def parse_model_id(model_id: str) -> tuple[str, str]:
        parts = model_id.rsplit("-", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (model_id, "")

    provider, version = parse_model_id("claude-3.5")
    print(f"  provider={provider!r}, version={version!r}")

    # Swap without temp variable
    a, b = 10, 20
    a, b = b, a
    print(f"  swapped: a={a}, b={b}")

    # Tuple of dicts — common for message lists
    messages = (
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    )
    print(f"  messages[1]['content']: {messages[1]['content']!r}")


# ── 6. Comprehensions ─────────────────────────────────────────────────────────
def demo_comprehensions():
    print("\n=== 6. Comprehensions ===")

    # List comprehension
    scores = [0.45, 0.88, 0.92, 0.61, 0.79, 0.95]
    high_scores = [s for s in scores if s >= 0.8]
    print(f"  high scores: {high_scores}")

    # Transform + filter in one step
    models = ["GPT-4O", "CLAUDE", "LLAMA3", "GEMINI"]
    clean  = [m.lower().replace("-", "_") for m in models if len(m) > 4]
    print(f"  cleaned:     {clean}")

    # Dict comprehension — batch score lookup
    model_names = ["gpt-4o", "claude", "llama3"]
    mock_scores = {name: round(0.7 + i * 0.1, 1) for i, name in enumerate(model_names)}
    print(f"  score dict:  {mock_scores}")

    # Set comprehension — unique domains from emails
    emails = ["alice@openai.com", "bob@anthropic.com", "carol@openai.com"]
    domains = {e.split("@")[1] for e in emails}
    print(f"  domains:     {domains}")

    # Generator expression (lazy — no list created)
    total_len = sum(len(e) for e in emails)
    print(f"  total chars: {total_len}")

    # Nested comprehension — build a token matrix
    matrix = [[f"t{r}{c}" for c in range(3)] for r in range(3)]
    for row in matrix:
        print(f"    {row}")


# ── 7. Practical AI pipeline patterns ─────────────────────────────────────────
def demo_ai_patterns():
    print("\n=== 7. Practical Patterns for AI Pipelines ===")

    # Accumulate message history
    def build_conversation(system: str, turns: list[tuple[str, str]]) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        for user_msg, assistant_msg in turns:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    history = build_conversation(
        "You are a Python tutor.",
        [
            ("What is a list?",   "A mutable ordered collection."),
            ("And a tuple?",      "An immutable ordered collection."),
        ],
    )
    print(f"  conversation ({len(history)} messages):")
    for m in history:
        print(f"    [{m['role']:<10}] {m['content']}")

    # Flatten nested results
    batches = [["chunk_1a", "chunk_1b"], ["chunk_2a"], ["chunk_3a", "chunk_3b", "chunk_3c"]]
    flat = [chunk for batch in batches for chunk in batch]
    print(f"\n  flattened chunks: {flat}")

    # Group results by key
    from collections import defaultdict
    items = [
        {"type": "tool_call",  "content": "search('python')"},
        {"type": "text",       "content": "Here's what I found:"},
        {"type": "tool_call",  "content": "calculator('2+2')"},
        {"type": "text",       "content": "The answer is 4."},
    ]
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[item["type"]].append(item["content"])
    print(f"\n  grouped by type: {dict(grouped)}")


if __name__ == "__main__":
    print("Python 01-Basics — Data Types & Structures")
    print("=" * 48)
    demo_scalar_types()
    demo_lists()
    demo_dicts()
    demo_sets()
    demo_tuples()
    demo_comprehensions()
    demo_ai_patterns()
