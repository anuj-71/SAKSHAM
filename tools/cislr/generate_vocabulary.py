"""
Generate gloss frequency table and recommended V1 vocabulary
without running MediaPipe (no cv2/mediapipe imports).
"""
import csv
from collections import Counter
from pathlib import Path

DATASET_CSV = Path("e:/gessture/cislr_data/dataset.csv")
LABEL_FREQ_CSV = Path("e:/gessture/cislr_data/label_frequency.csv")


def normalize_label(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().upper()


def main(limit=0, vocab_size=40):
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    freq = Counter()
    for r in rows:
        gloss = normalize_label(r.get("gloss", r.get("label", "")))
        freq[gloss] += 1

    # write freq CSV
    with open(LABEL_FREQ_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gloss", "count"])
        for g, c in freq.most_common():
            writer.writerow([g, c])

    recommended = [g for g, _ in freq.most_common(vocab_size)]

    out = Path("e:/gessture/cislr_data/landmarks/recommended_v1.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for item in recommended:
            f.write(item + "\n")

    print("Wrote", LABEL_FREQ_CSV)
    print("Wrote", out)


if __name__ == "__main__":
    main()
