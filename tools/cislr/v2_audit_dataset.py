"""
V2.0 Dataset Audit: Hand-presence statistics per gloss.

Scans ALL accepted NPZ files and computes per-gloss:
  - Total samples
  - Samples with both hands >50% of frames
  - Samples with one hand >50% of frames
  - Samples with neither hand consistently present
  - Mean left/right presence ratio

Writes reports to cislr_data/reports/v2_audit_{json,md,csv}
"""
from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ACCEPTED_DIR = Path("cislr_data/landmarks/accepted")
REPORTS_DIR = Path("cislr_data/reports")
PRESENCE_THRESHOLD = 0.5  # fraction of frames with hand detection


def classify_presence(left_present: np.ndarray, right_present: np.ndarray) -> str:
    lp = float(np.mean(left_present))
    rp = float(np.mean(right_present))
    if lp > PRESENCE_THRESHOLD and rp > PRESENCE_THRESHOLD:
        return "both"
    elif lp > PRESENCE_THRESHOLD or rp > PRESENCE_THRESHOLD:
        return "one"
    else:
        return "none"


def main() -> int:
    npz_files = sorted(ACCEPTED_DIR.glob("*.npz"))
    logging.info("Found %d NPZ files to audit", len(npz_files))

    # Per-gloss accumulators
    gloss_stats: dict[str, dict] = {}

    for idx, npz_path in enumerate(npz_files):
        if (idx + 1) % 1000 == 0:
            logging.info("Progress: %d / %d", idx + 1, len(npz_files))

        try:
            with np.load(npz_path) as data:
                gloss = str(data["gloss"])
                lp_arr = np.asarray(data["left_present"], dtype=np.float32)
                rp_arr = np.asarray(data["right_present"], dtype=np.float32)
                n_frames = int(data["frame_count"])
        except Exception as e:
            logging.warning("Failed to load %s: %s", npz_path.name, e)
            continue

        cat = classify_presence(lp_arr, rp_arr)
        lp_mean = float(np.mean(lp_arr))
        rp_mean = float(np.mean(rp_arr))

        if gloss not in gloss_stats:
            gloss_stats[gloss] = {
                "gloss": gloss,
                "total": 0,
                "both": 0,
                "one": 0,
                "none": 0,
                "sum_lp": 0.0,
                "sum_rp": 0.0,
                "min_frames": n_frames,
                "max_frames": n_frames,
                "sum_frames": 0,
            }
        gs = gloss_stats[gloss]
        gs["total"] += 1
        gs[cat] += 1
        gs["sum_lp"] += lp_mean
        gs["sum_rp"] += rp_mean
        gs["min_frames"] = min(gs["min_frames"], n_frames)
        gs["max_frames"] = max(gs["max_frames"], n_frames)
        gs["sum_frames"] += n_frames

    # Build ranked list
    ranked = []
    for gs in gloss_stats.values():
        usable = gs["both"] + gs["one"]
        ranked.append({
            "gloss": gs["gloss"],
            "total": gs["total"],
            "both": gs["both"],
            "one": gs["one"],
            "none": gs["none"],
            "usable_pct": round(usable / gs["total"] * 100, 1) if gs["total"] else 0.0,
            "usable_count": usable,
            "mean_lp": round(gs["sum_lp"] / gs["total"], 3),
            "mean_rp": round(gs["sum_rp"] / gs["total"], 3),
            "min_frames": gs["min_frames"],
            "max_frames": gs["max_frames"],
            "mean_frames": round(gs["sum_frames"] / gs["total"], 1),
        })

    ranked.sort(key=lambda r: (-r["usable_count"], -r["usable_pct"], r["gloss"]))

    # Summary
    total_samples = len(npz_files)
    total_glosses = len(ranked)

    both_total = sum(r["both"] for r in ranked)
    one_total = sum(r["one"] for r in ranked)
    none_total = sum(r["none"] for r in ranked)

    # Find V2 vocabulary candidates
    v2_candidates = [r for r in ranked if r["usable_count"] >= 10]
    v2_high = [r for r in ranked if r["usable_count"] >= 10 and r["usable_pct"] >= 50.0]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "total_accepted_npz": total_samples,
        "total_unique_glosses": total_glosses,
        "hand_presence_summary": {
            "both_hands": f"{both_total} ({both_total/total_samples*100:.1f}%)",
            "one_hand": f"{one_total} ({one_total/total_samples*100:.1f}%)",
            "neither": f"{none_total} ({none_total/total_samples*100:.1f}%)",
        },
        "vocabulary_candidates": {
            "usable_count>=10": len(v2_candidates),
            "usable_count>=10_and_usable_pct>=50": len(v2_high),
        },
        "gloss_rankings": ranked,
        "recommended_v2": v2_high[:15],
    }

    # Write JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "v2_audit.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote %s", json_path)

    # Write CSV
    csv_path = REPORTS_DIR / "v2_audit.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "gloss", "total", "both", "one", "none",
            "usable_count", "usable_pct",
            "mean_lp", "mean_rp",
            "min_frames", "max_frames", "mean_frames",
        ])
        w.writeheader()
        for r in ranked:
            w.writerow({k: r[k] for k in w.fieldnames})
    logging.info("Wrote %s", csv_path)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  V2.0 DATASET AUDIT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total NPZ files:     {total_samples}")
    print(f"  Total unique glosses: {total_glosses}")
    print(f"  Both hands (>50%):   {both_total:>5d} ({both_total/total_samples*100:.1f}%)")
    print(f"  One hand  (>50%):    {one_total:>5d} ({one_total/total_samples*100:.1f}%)")
    print(f"  Neither hand:        {none_total:>5d} ({none_total/total_samples*100:.1f}%)")
    print(f"\n  Glosses with >=10 usable samples:        {len(v2_candidates)}")
    print(f"  Glosses with >=10 usable AND >=50% usable: {len(v2_high)}")
    print(f"\n  Top 15 recommended V2 candidates:")
    print(f"  {'Gloss':30s} {'Total':>6s} {'Both':>5s} {'One':>5s} {'Usable%':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*5} {'-'*8}")
    for r in v2_high[:15]:
        print(f"  {r['gloss']:30s} {r['total']:6d} {r['both']:5d} {r['one']:5d} {r['usable_pct']:7.1f}%")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
