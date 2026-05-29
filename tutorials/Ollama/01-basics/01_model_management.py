"""
Ollama 01-Basics — Model Management
=====================================
Topics covered:
  1. List locally installed models
  2. Pull (download) a model
  3. Inspect model info (parameters, quantisation, context length)
  4. Check if a model is available before using it
  5. Delete a model

Prerequisites:
  - Ollama installed and running: `ollama serve`
  - At least one model pulled: `ollama pull llama3.2`

Run:
  python 01_model_management.py
"""

import os
from dotenv import load_dotenv
import ollama

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ── 1. List installed models ───────────────────────────────────────────────────
def demo_list_models():
    print("\n=== 1. List Installed Models ===")
    response = ollama.list()
    models = response.models if hasattr(response, "models") else response.get("models", [])

    if not models:
        print("  No models installed yet. Run: ollama pull llama3.2")
        return

    print(f"  {'NAME':<30} {'SIZE':>10}  {'MODIFIED'}")
    print(f"  {'-'*30} {'-'*10}  {'-'*20}")
    for m in models:
        name = m.model if hasattr(m, "model") else m.get("name", "?")
        size = m.size if hasattr(m, "size") else m.get("size", 0)
        modified = str(m.modified_at if hasattr(m, "modified_at") else m.get("modified_at", ""))[:10]
        size_gb = size / 1e9 if isinstance(size, (int, float)) else 0
        print(f"  {name:<30} {size_gb:>9.2f}G  {modified}")


# ── 2. Pull a model ────────────────────────────────────────────────────────────
def demo_pull_model(model_name: str = "llama3.2:1b"):
    print(f"\n=== 2. Pull Model: {model_name} ===")
    print(f"  Pulling '{model_name}' (this downloads if not cached)...")

    # pull() streams progress events
    last_status = ""
    for progress in ollama.pull(model_name, stream=True):
        status = progress.status if hasattr(progress, "status") else progress.get("status", "")
        if status and status != last_status:
            print(f"  [{status}]")
            last_status = status

    print(f"  Done — '{model_name}' is ready.")


# ── 3. Inspect model info ──────────────────────────────────────────────────────
def demo_model_info(model_name: str = MODEL):
    print(f"\n=== 3. Model Info: {model_name} ===")
    try:
        info = ollama.show(model_name)
    except ollama.ResponseError as e:
        print(f"  Model not found: {e}. Pull it first with: ollama pull {model_name}")
        return

    # Model file / template
    modelfile = info.modelfile if hasattr(info, "modelfile") else info.get("modelfile", "")
    template  = info.template  if hasattr(info, "template")  else info.get("template", "")

    # Details (parameters, quantisation, etc.)
    details = info.details if hasattr(info, "details") else info.get("details", {})
    if hasattr(details, "__dict__"):
        details = details.__dict__

    print(f"  Model:           {model_name}")
    print(f"  Family:          {details.get('family', 'n/a')}")
    print(f"  Parameter size:  {details.get('parameter_size', 'n/a')}")
    print(f"  Quantisation:    {details.get('quantization_level', 'n/a')}")
    print(f"  Context length:  {details.get('context_length', 'n/a')}")
    print(f"  Template present: {'yes' if template else 'no'}")

    # Show first few lines of the modelfile
    if modelfile:
        lines = [l for l in modelfile.splitlines() if l.strip()][:6]
        print(f"\n  Modelfile (first 6 lines):")
        for line in lines:
            print(f"    {line}")


# ── 4. Check model availability ───────────────────────────────────────────────
def demo_check_availability():
    print("\n=== 4. Check Model Availability ===")
    models_to_check = ["llama3.2", "llama3.2:1b", "mistral", "codellama", "llava", "nomic-embed-text"]

    response = ollama.list()
    installed = {
        (m.model if hasattr(m, "model") else m.get("name", ""))
        for m in (response.models if hasattr(response, "models") else response.get("models", []))
    }

    for name in models_to_check:
        # Match by prefix (e.g. "llama3.2" matches "llama3.2:latest")
        available = any(i == name or i.startswith(f"{name}:") for i in installed)
        status = "✓ installed" if available else "✗ not installed"
        print(f"  {name:<25} {status}")


# ── 5. Delete a model (safe demo — only deletes a small test model) ───────────
def demo_delete_model():
    print("\n=== 5. Delete a Model ===")
    # We only demonstrate the API without actually deleting a useful model.
    # Uncomment the lines below to delete a specific model.
    print("  ollama.delete() removes a model from local storage.")
    print("  Example (not executed here to preserve your models):")
    print("    ollama.delete('llama3.2:1b')")
    print()
    print("  Or from the CLI:")
    print("    ollama rm llama3.2:1b")


if __name__ == "__main__":
    print("Ollama 01-Basics — Model Management")
    print("=" * 45)
    print("Requires Ollama running locally: ollama serve")

    if not check_ollama_running():
        print("\n  ERROR: Cannot connect to Ollama at http://localhost:11434")
        print("  Start the server with: ollama serve")
        raise SystemExit(1)

    print("  Ollama server: OK\n")

    demo_list_models()
    demo_model_info()
    demo_check_availability()
    demo_delete_model()
    # demo_pull_model("llama3.2:1b")  # Uncomment to actually pull a model
