"""Analyze v2 audit results."""
import json

with open('cislr_data/reports/v2_audit.json') as f:
    r = json.load(f)

ranked = r['gloss_rankings']

print(f"Total glosses with usable > 0: {sum(1 for g in ranked if g['usable_count'] > 0)}")
print(f"Total glosses with usable == 0: {sum(1 for g in ranked if g['usable_count'] == 0)}")
print()

for threshold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    count = sum(1 for g in ranked if g['usable_count'] >= threshold)
    print(f"  usable_count >= {threshold}: {count} glosses")

print()
print(f"{'Gloss':30s} {'Total':>5s} {'Both':>5s} {'One':>5s} {'Usable':>7s} {'Usable%':>8s}")
print("-"*60)
for g in ranked[:30]:
    print(f"{g['gloss']:30s} {g['total']:5d} {g['both']:5d} {g['one']:5d} {g['usable_count']:7d} {g['usable_pct']:7.1f}%")

print()
# find best usable_pct with decent count
print("Glosses with usable_pct >= 30% AND usable_count >= 4:")
print(f"{'Gloss':30s} {'Total':>5s} {'Both':>5s} {'One':>5s} {'Usable':>7s} {'Usable%':>8s}")
print("-"*60)
candidates = [g for g in ranked if g['usable_pct'] >= 30.0 and g['usable_count'] >= 4]
for g in candidates:
    print(f"{g['gloss']:30s} {g['total']:5d} {g['both']:5d} {g['one']:5d} {g['usable_count']:7d} {g['usable_pct']:7.1f}%")
print(f"Total such glosses: {len(candidates)}")

# What about looser threshold? usable_pct >= 20% and usable >= 5
print()
print("Glosses with usable_pct >= 20% AND usable_count >= 5:")
candidates2 = [g for g in ranked if g['usable_pct'] >= 20.0 and g['usable_count'] >= 5]
for g in candidates2:
    print(f"{g['gloss']:30s} {g['total']:5d} {g['both']:5d} {g['one']:5d} {g['usable_count']:7d} {g['usable_pct']:7.1f}%")
print(f"Total such glosses: {len(candidates2)}")
