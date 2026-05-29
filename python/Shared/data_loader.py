"""
Shared data loading helpers.

Provides thin wrappers around the standard library for reading
CSV files (as list-of-dicts) and JSON files.
"""
import csv
import json


def load_csv(filepath) -> list:
    """Read all rows from a CSV file and return a list of dicts."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_json(filepath):
    """Load and return the parsed contents of a JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
