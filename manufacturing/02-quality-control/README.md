# 02 — Quality Control

> Use AI to review product inspection reports and decide whether a product passes or fails quality checks.

---

## What This Does

Quality control (QC) is critical in manufacturing. Inspectors record measurements and observations
for every product batch. Reviewing them manually is slow and inconsistent.

This script feeds each inspection record to an AI model. The AI returns:
- A **pass / fail** verdict
- A **confidence score** (0–100 %)
- A clear **reason** for the decision
- Specific **issues found** (if any)

---

## How to Run

```bash
cd manufacturing/02-quality-control
python quality_control.py
```

---

## Expected Input

File: `data/sample_inspections.json`

Each record has:
- `batch_id` — unique batch identifier
- `product` — product name
- `measurements` — dict of measured values vs. spec tolerances
- `visual_notes` — text notes from the inspector

---

## Sample Output

```
🔍 Quality Control Inspector
   AI Provider: OPENAI
============================================================

📦 Batch: BATCH-003  |  Product: Valve Housing
   Verdict    : ❌ FAIL  (confidence: 94%)
   Issues     : Wall thickness 2.1mm (spec: 2.5±0.2mm). Surface scratch detected.
   Action     : Reject batch. Review moulding die for wear.

📦 Batch: BATCH-005  |  Product: Gear Assembly
   Verdict    : ✅ PASS  (confidence: 98%)
   Issues     : None
   Action     : Approve for shipment.

✅ Inspection complete. 3/5 batches passed.
```

---

## How It Works

1. Load inspection records from JSON
2. Format each record into a detailed AI prompt
3. Ask the AI to evaluate against specifications
4. Parse the structured JSON response
5. Display a clear pass/fail summary

---

## Extending This Use-Case

- Connect to a camera feed and send image descriptions for visual inspection
- Add industry-specific tolerances (ISO, DIN standards)
- Export a QC report to PDF or Excel
