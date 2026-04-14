"""
workout_parser.py
-----------------
X elemt_of (A, B, C)
Parses fitness log files (W-NNN-X.txt) into a structured dictionary.
    

Output schema
─────────────
{
  "W-169-A": {
    "filename": "W-169-A.txt",
    "day_number": 169,
    "session_type": "A",
    "date": datetime(2023, 1, 21),
    "date_str": "2023-01-21",
    "body_weight_kg": None,          # float or None
    "kcal_eaten": None,              # float or None
    "exercises": [
      {
        "name": "bench_press",
        "sets": [
          {
            "weight_kg": 72.5,       # float or None (None when bodyweight)
            "bodyweight": False,     # True when weight is 'bw'
            "reps": [5, 5, 5, 5, 5], # list[int]
          },
          ...
        ],
        "cardio_minutes": None,      # int, only for treadmill-style entries
      },
      ...
    ],
    "parse_warnings": [],            # list of strings for anything fuzzy
  },
  ...
}
"""

import os
import re
from datetime import datetime
from pathlib import Path
from pprint import pformat

import heatmap


_DATE_FORMATS = [
    # "21Jan2023"  /  "3Feb2023"
    ("%d%b%Y", re.compile(r"^\d{1,2}[A-Za-z]{3}\d{4}$")),
    # "2023-03-25"
    ("%Y-%m-%d", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    # "21/01/2023"  (just in case)
    ("%d/%m/%Y", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
]


def _parse_date(token: str):
    """
        Return (datetime, date_str) or (None, None) if unrecognised
    """
    t = token.strip()
    for fmt, pat in _DATE_FORMATS:
        if pat.match(t):
            try:
                dt = datetime.strptime(t, fmt)
                return dt, dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    return None, None



def _parse_weight(raw: str):
    """
    Parse a weight token like '72.5 kg', '72.5kg', 'bw', '0 kg' …
    Returns (weight_kg: float | None, is_bodyweight: bool).
    weight_kg is None only when is_bodyweight is True.
    """
    raw = raw.strip().lower()
    if raw in ("bw", "bodyweight"):
        return None, True
    # strip unit, allow optional space
    m = re.match(r"([\d.]+)\s*kg?", raw)
    if m:
        return float(m.group(1)), False
    return None, False



def _parse_reps(raw: str):
    """
    Parse a reps token.  Supported forms:
      "5s5r"      → 5 sets of 5 reps  → [5,5,5,5,5]
      "5s10r"     → [10,10,10,10,10]
      "3s24r"     → [24,24,24]
      "8,8,7,6,6" → [8,8,7,6,6]
      "10,10,8,8,8" → [10,10,8,8,8]
    Returns list[int] or [] on failure.
    """
    raw = raw.strip().rstrip("r").rstrip(";")

    # "NsNr" or "NsN"
    m = re.match(r"(\d+)s(\d+)r?$", raw)
    if m:
        sets, reps = int(m.group(1)), int(m.group(2))
        return [reps] * sets

    # separated by comma "8,8,7,6,6"
    if re.match(r"[\d,]+$", raw):
        return [int(x) for x in raw.split(",") if x]

    return []


def _parse_set_line(line: str):
    """
        Parse a data line inside an exercise block.
        Examples
            "w 72.5 kg; rps 5s5r;"
            "w 20 kg; rps 2s12r;"
            "w 35kg; rps 5s8r;"
            "w 72.5 kg; rps; 1s5r;"     ← extra semicolon after rps
            "w 200kg; 1s6r;"             ← missing 'rps' keyword
            "w 0 kg;"                    ← weight-only (bw block)
            "rps 8,8,7,6,6;"
            "mins 15;"
        Returns one of:
            ("set",     {weight_kg, bodyweight, reps})
            ("cardio",  {minutes})
            ("kcal",    {kcal})
            ("unknown", raw_line)
    """
    line = line.rstrip(";").strip()

    # cardio / treadmill
    m = re.match(r"mins\s+(\d+)", line, re.IGNORECASE)
    if m:
        return "cardio", {"minutes": int(m.group(1))}

    # energy  "e 0 kcal"
    m = re.match(r"e\s+([\d.]+)\s*kcal", line, re.IGNORECASE)
    if m:
        return "kcal", {"kcal": float(m.group(1))}

    # weight + reps  "w <weight>; rps <reps>"  (possibly "rps; <reps>")
    # Also handles "w <weight>; <reps>" (missing rps keyword)
    m = re.match(r"w\s+(.+?);\s*(?:rps;?\s*)(.+)", line, re.IGNORECASE)
    if m:
        weight_kg, is_bw = _parse_weight(m.group(1))
        reps = _parse_reps(m.group(2))
        return "set", {"weight_kg": weight_kg, "bodyweight": is_bw, "reps": reps}

    # weight only, no reps  "w 0 kg"  (used in bw block with 0 kg)
    m = re.match(r"^w\s+(.+)$", line, re.IGNORECASE)
    if m:
        weight_kg, is_bw = _parse_weight(m.group(1))
        return "set", {"weight_kg": weight_kg, "bodyweight": is_bw, "reps": []}

    # reps only  "rps 8,8,7,6,6"
    m = re.match(r"rps\s+(.+)", line, re.IGNORECASE)
    if m:
        reps = _parse_reps(m.group(1))
        return "set", {"weight_kg": None, "bodyweight": True, "reps": reps}

    return "unknown", line


def _normalise_exercise_name(raw: str) -> str:
    """Strip trailing colon/semicolons and lower-case the exercise name."""
    return raw.strip().rstrip(":;").strip().lower()


# Main file parser
_FILENAME_RE = re.compile(r"^W-(\d+)-([A-Za-z])\.txt$", re.IGNORECASE)

# These are meta-keys, not exercises
_META_KEYS = {"bw", "kcal"}


def parse_workout_file(filepath: str | Path) -> dict:
    """Parse one workout log file and return its structured dict."""
    path = Path(filepath)
    fname = path.name

    # Extract day number and session type from filename
    m = _FILENAME_RE.match(fname)
    day_number = int(m.group(1)) if m else None
    session_type = m.group(2).upper() if m else None

    warnings = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()

    lines = [l.rstrip("\n") for l in raw_lines]

    # Pass 1: find the date (first non-empty line that looks like a date)
    date_obj, date_str = None, None
    date_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        date_obj, date_str = _parse_date(stripped)
        if date_obj:
            date_line_idx = i
            break
        # first non-empty line that is NOT a date → warn and move on
        warnings.append(f"Could not parse date from first non-empty line: {repr(stripped)}")
        break

    # Pass 2: walk remaining lines to collect exercise blocks
    body_weight_kg: float | None = None
    kcal_eaten: float | None = None
    exercises: list[dict] = []

    current_exercise: str | None = None
    current_sets: list[dict] = []
    current_cardio: int | None = None

    def flush_exercise():
        nonlocal current_exercise, current_sets, current_cardio
        if current_exercise is None:
            return
        if current_exercise not in _META_KEYS:
            # Merge rows that share the same weight key into one entry,
            # preserving the original ordering of first appearance.
            merged = {}
            for s in current_sets:
                key = (s["weight_kg"], s["bodyweight"])
                if key in merged:
                    merged[key]["reps"].extend(s["reps"])
                else:
                    merged[key] = {
                        "weight_kg": s["weight_kg"],
                        "bodyweight": s["bodyweight"],
                        "reps": list(s["reps"]),
                    }
            exercises.append({
                "name": current_exercise,
                "sets": list(merged.values()),
                "cardio_minutes": current_cardio,
            })
        current_exercise = None
        current_sets = []
        current_cardio = None

    start = (date_line_idx + 1) if date_line_idx is not None else 0

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue

        # Exercise header line: ends with ":" or ";" (or neither, just the name) ──
        # Heuristic: no leading spaces AND matches exercise-name pattern
        is_header = (
            not line.startswith((" ", "\t"))
            and re.match(r"^[a-z_]+[;:]?\s*$", stripped, re.IGNORECASE)
        )

        if is_header:
            flush_exercise()
            current_exercise = _normalise_exercise_name(stripped)
            continue

        # ── Data line inside a block ──
        if current_exercise is None:
            warnings.append(f"Orphan data line (no exercise header): {repr(stripped)}")
            continue

        kind, parsed = _parse_set_line(stripped)

        if kind == "set":
            if current_exercise == "bw":
                # body weight entry; 0 means not tracked that day
                bw = parsed["weight_kg"]
                body_weight_kg = bw if (bw is not None and bw != 0.0) else body_weight_kg
            elif current_exercise == "kcal":
                pass  # handled via "kcal" kind below
            else:
                current_sets.append(parsed)

        elif kind == "kcal":
            kcal_eaten = parsed["kcal"]

        elif kind == "cardio":
            current_cardio = parsed["minutes"]

        elif kind == "unknown":
            # Could be "e 0 kcal" under kcal block
            m_e = re.match(r"e\s+([\d.]+)\s*kcal", stripped, re.IGNORECASE)
            if m_e and current_exercise == "kcal":
                kcal_eaten = float(m_e.group(1))
            else:
                warnings.append(f"Unrecognised line under '{current_exercise}': {repr(stripped)}")

    flush_exercise()

    return {
        "filename": fname,
        "day_number": day_number,
        "session_type": session_type,
        "date": date_obj,
        "date_str": date_str,
        "body_weight_kg": body_weight_kg,
        "kcal_eaten": kcal_eaten,
        "exercises": exercises,
        "parse_warnings": warnings,
    }


def load_workout_directory(directory: str):
    """
        Parse all W-NNN-X.txt files in *directory*.
        Returns an OrderedDict keyed by "W-NNN-X" (filename without extension),
        sorted by day number then session type
    """
    directory = Path(directory)
    files = sorted(
        directory.glob("W-*-*.txt"),
        key=lambda p: (
            int(m.group(1)) if (m := _FILENAME_RE.match(p.name)) else 9999,
            m.group(2).upper() if m else "Z",
        ),
    )

    result = {}
    for f in files:
        key = f.stem  # "W-169-A"
        result[key] = parse_workout_file(f)

    return result