"""
Ollama 03-Advanced — Vision Models (llava)
==========================================
Topics covered:
  1. Sending a local image file to llava
  2. Sending a base64-encoded image
  3. Image description / captioning
  4. Visual question answering (VQA)
  5. Multi-turn conversation about an image

Prerequisites:
  - Ollama running: `ollama serve`
  - Vision model pulled: `ollama pull llava`
  - Optional: a local image file (PNG/JPG) to test with

Run:
  python 01_vision_models.py
"""

import os
import sys
import base64
import struct
import zlib
from pathlib import Path
from dotenv import load_dotenv
import ollama

load_dotenv()

VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
BASE_URL     = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")


# ── Helper: create a minimal synthetic PNG for demos ─────────────────────────
def create_test_png(path: Path, label: str = "AI"):
    """Write a tiny valid 16x16 PNG with a coloured rectangle and text label."""
    import struct, zlib

    width, height = 64, 64
    # Simple gradient: red→green pixels
    raw_rows = []
    for y in range(height):
        row = b"\x00"  # filter type None
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            row += bytes([r, g, b])
        raw_rows.append(row)

    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data)

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr_data)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def check_vision_model() -> bool:
    try:
        response = ollama.list()
        models = response.models if hasattr(response, "models") else response.get("models", [])
        installed = {
            (m.model if hasattr(m, "model") else m.get("name", ""))
            for m in models
        }
        return any(VISION_MODEL in name for name in installed)
    except Exception:
        return False


# ── 1. Basic image description ────────────────────────────────────────────────
def demo_image_description(image_path: Path):
    print("\n=== 1. Image Description / Captioning ===")
    print(f"  Image: {image_path}")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Describe this image in detail. What colours, shapes, and patterns do you see?",
                "images": [str(image_path)],
            }
        ],
    )
    description = response.message.content if hasattr(response, "message") else response["message"]["content"]
    print(f"  Description: {description.strip()[:400]}")


# ── 2. Image from base64 ──────────────────────────────────────────────────────
def demo_base64_image(image_path: Path):
    print("\n=== 2. Image via Base64 Encoding ===")
    b64 = image_to_base64(image_path)
    print(f"  Base64 length: {len(b64)} chars (first 40: {b64[:40]}...)")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is the dominant colour in this image?",
                "images": [b64],
            }
        ],
    )
    answer = response.message.content if hasattr(response, "message") else response["message"]["content"]
    print(f"  Answer: {answer.strip()}")


# ── 3. Visual question answering ──────────────────────────────────────────────
def demo_vqa(image_path: Path):
    print("\n=== 3. Visual Question Answering ===")
    questions = [
        "How many distinct colour regions are visible?",
        "Does this image look like it was generated programmatically or is it a photograph?",
        "What would this pattern look like if it were a real-world material?",
    ]

    for q in questions:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": q, "images": [str(image_path)]}],
        )
        answer = response.message.content if hasattr(response, "message") else response["message"]["content"]
        print(f"\n  Q: {q}")
        print(f"  A: {answer.strip()[:200]}")


# ── 4. Multi-turn conversation about an image ─────────────────────────────────
def demo_multi_turn_vision(image_path: Path):
    print("\n=== 4. Multi-Turn Conversation About an Image ===")

    # First turn: send the image
    history = [
        {
            "role": "user",
            "content": "I'm going to ask you several questions about this image. "
                       "First, please give me a one-sentence overview.",
            "images": [str(image_path)],
        }
    ]

    response = ollama.chat(model=VISION_MODEL, messages=history)
    msg = response.message.content if hasattr(response, "message") else response["message"]["content"]
    history.append({"role": "assistant", "content": msg})
    print(f"  Turn 1 — Overview: {msg.strip()[:200]}")

    # Follow-up turns (no need to resend image)
    follow_ups = [
        "What colours transition from left to right?",
        "If this were a logo, what industry might it represent?",
    ]
    for question in follow_ups:
        history.append({"role": "user", "content": question})
        response = ollama.chat(model=VISION_MODEL, messages=history)
        answer = response.message.content if hasattr(response, "message") else response["message"]["content"]
        history.append({"role": "assistant", "content": answer})
        print(f"\n  Q: {question}")
        print(f"  A: {answer.strip()[:200]}")


# ── 5. Use a custom image path ────────────────────────────────────────────────
def demo_custom_image():
    print("\n=== 5. Using Your Own Image ===")
    print("  To use your own image, modify this file and set:")
    print("    image_path = Path('/path/to/your/image.jpg')")
    print("  Then pass it to any of the demo functions above.")
    print("  Supported formats: PNG, JPEG, GIF, WebP")
    print()
    print("  Example:")
    print("    image_path = Path('photo.jpg')")
    print("    demo_image_description(image_path)")
    print("    demo_vqa(image_path)")


if __name__ == "__main__":
    print("Ollama 03-Advanced — Vision Models")
    print("=" * 45)
    print(f"  Vision model: {VISION_MODEL}")
    print("  Requires: ollama pull llava\n")

    try:
        import ollama as _test
        _test.list()
    except Exception:
        print("  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    if not check_vision_model():
        print(f"  WARNING: '{VISION_MODEL}' is not installed.")
        print(f"  Pull it with: ollama pull {VISION_MODEL}")
        print("  Demos will be skipped.\n")
        demo_custom_image()
        sys.exit(0)

    # Create a synthetic test image in a temp location
    test_image = Path(__file__).parent / "_test_gradient.png"
    create_test_png(test_image)
    print(f"  Created synthetic test image: {test_image}\n")

    demo_image_description(test_image)
    demo_base64_image(test_image)
    demo_vqa(test_image)
    demo_multi_turn_vision(test_image)
    demo_custom_image()

    # Clean up
    if test_image.exists():
        test_image.unlink()
        print(f"\n  Cleaned up test image: {test_image}")
