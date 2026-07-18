import argparse
import csv
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_LABELS = [
    "SEE YOU TOMORROW",
    "HELP",
    "STOP",
    "YES",
    "NO",
    "WATER",
    "THANK YOU",
    "PLEASE",
    "SORRY",
    "NO_SIGN",
]

LABEL_COLUMN_CANDIDATES = (
    "label",
    "gloss",
    "word",
    "target",
    "class",
    "name",
    "text",
    "english",
    "translation",
)

_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[_\-]+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^A-Z0-9 ]+")

ALIASES = {
    "HELLO": "SEE YOU TOMORROW",
    "HELP": "HELP",
    "STOP": "STOP",
    "YES": "YES",
    "NO": "NO",
    "WATER": "WATER",
    "THANK YOU": "THANK YOU",
    "THANKYOU": "THANK YOU",
    "THANKS": "THANK YOU",
    "PLEASE": "PLEASE",
    "SORRY": "SORRY",
    "NO SIGN": "NO_SIGN",
    "NO_SIGN": "NO_SIGN",
}


def canonicalize_label(raw_label: str) -> str:
    text = str(raw_label or "").strip().upper()
    text = _SEPARATOR_RE.sub(" ", text)
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def map_to_saksham_label(raw_label: str) -> Optional[str]:
    normalized = canonicalize_label(raw_label)
    return ALIASES.get(normalized)


def detect_label_column(fieldnames: Sequence[str], preferred: Optional[str] = None) -> str:
    if preferred:
        if preferred not in fieldnames:
            raise ValueError(f"Requested label column '{preferred}' not found. Available columns: {fieldnames}")
        return preferred

    lowered = {name.lower(): name for name in fieldnames}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    raise ValueError(
        "Unable to infer the label column. Use --label-column to specify it explicitly. "
        f"Available columns: {fieldnames}"
    )


def load_csv_rows(input_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(input_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {input_path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def normalize_rows(
    rows: Iterable[Dict[str, str]],
    label_column: str,
) -> List[Dict[str, str]]:
    normalized_rows: List[Dict[str, str]] = []
    for row in rows:
        raw_label = row.get(label_column, "")
        normalized_label = canonicalize_label(raw_label)
        saksham_label = map_to_saksham_label(raw_label)
        mapping_status = "mapped" if saksham_label else "unmapped"

        normalized_row = dict(row)
        normalized_row["raw_label"] = raw_label
        normalized_row["normalized_label"] = normalized_label
        normalized_row["saksham_label"] = saksham_label or ""
        normalized_row["mapping_status"] = mapping_status
        normalized_rows.append(normalized_row)
    return normalized_rows


def write_normalized_csv(output_path: str, rows: Sequence[Dict[str, str]], original_fieldnames: Sequence[str]) -> None:
    extra_fields = ["raw_label", "normalized_label", "saksham_label", "mapping_status"]
    fieldnames = list(original_fieldnames)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_normalization(rows: Sequence[Dict[str, str]]) -> Dict[str, Counter]:
    saksham_counts = Counter()
    normalized_counts = Counter()
    unmapped_counts = Counter()

    for row in rows:
        normalized = row.get("normalized_label", "")
        normalized_counts[normalized] += 1
        saksham_label = row.get("saksham_label", "")
        if saksham_label:
            saksham_counts[saksham_label] += 1
        else:
            unmapped_counts[normalized] += 1

    return {
        "saksham_counts": saksham_counts,
        "normalized_counts": normalized_counts,
        "unmapped_counts": unmapped_counts,
    }


def build_default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_normalized{ext or '.csv'}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize external sign-language labels into SAKSHAM vocabulary.")
    parser.add_argument("--input", required=True, help="Path to the source CSV metadata file.")
    parser.add_argument("--output", help="Path to write the normalized CSV. Defaults to <input>_normalized.csv")
    parser.add_argument("--label-column", help="Optional explicit label/gloss column name.")
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="How many unmapped normalized labels to print in the summary.",
    )
    args = parser.parse_args()

    fieldnames, rows = load_csv_rows(args.input)
    label_column = detect_label_column(fieldnames, preferred=args.label_column)
    normalized_rows = normalize_rows(rows, label_column=label_column)

    output_path = args.output or build_default_output_path(args.input)
    write_normalized_csv(output_path, normalized_rows, original_fieldnames=fieldnames)

    summary = summarize_normalization(normalized_rows)
    print(f"Input file: {args.input}")
    print(f"Detected label column: {label_column}")
    print(f"Rows processed: {len(normalized_rows)}")
    print(f"Normalized CSV: {output_path}")
    print("Mapped SAKSHAM label counts:")
    for label in TARGET_LABELS:
        print(f"- {label}: {summary['saksham_counts'].get(label, 0)}")

    if summary["unmapped_counts"]:
        print("Top unmapped normalized labels:")
        for label, count in summary["unmapped_counts"].most_common(max(args.preview_limit, 0)):
            display_label = label or "<empty>"
            print(f"- {display_label}: {count}")
    else:
        print("All labels mapped successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
