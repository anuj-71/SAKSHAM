"""Explore relaxed presence thresholds for V2 vocabulary."""
import json

with open('cislr_data/reports/v2_audit.json') as f:
    r = json.load(f)

ranked = r['gloss_rankings']

# --- Check with different presence thresholds ---
# We need to re-analyze because the threshold was baked into the script.
# For now, let's examine the raw data differently:
# Look for glosses where sum(both + one) >= 5 OR total samples >= 10

print("=== OPTION A: glosses with total >= 10 samples ===")
candidates_a = [g for g in ranked if g['total'] >= 10]
print(f"Count: {len(candidates_a)}")
for g in candidates_a:
    print(f"  {g['gloss']:30s} total={g['total']:2d} both={g['both']:2d} one={g['one']:2d} none={g['none']:2d} usable%={g['usable_pct']:5.1f}%")

print()
print("=== OPTION B: glosses with usable >= 3 (any percentage) ===")
candidates_b = [g for g in ranked if g['usable_count'] >= 3]
print(f"Count: {len(candidates_b)}")
for g in candidates_b:
    print(f"  {g['gloss']:30s} total={g['total']:2d} both={g['both']:2d} one={g['one']:2d} none={g['none']:2d} usable%={g['usable_pct']:5.1f}%")

print()
print("=== OPTION C: glosses with usable >= 2 AND total >= 10 ===")
candidates_c = [g for g in ranked if g['usable_count'] >= 2 and g['total'] >= 10]
print(f"Count: {len(candidates_c)}")
for g in candidates_c:
    print(f"  {g['gloss']:30s} total={g['total']:2d} both={g['both']:2d} one={g['one']:2d} none={g['none']:2d} usable%={g['usable_pct']:5.1f}%")

print()
print("=== OPTION D: ANY gloss with usable > 0 AND total >= 5 ===")
candidates_d = [g for g in ranked if g['usable_count'] > 0 and g['total'] >= 5]
print(f"Count: {len(candidates_d)}")
for g in candidates_d:
    print(f"  {g['gloss']:30s} total={g['total']:2d} both={g['both']:2d} one={g['one']:2d} none={g['none']:2d} usable%={g['usable_pct']:5.1f}%")

# Also check: do the V1's best classes (MONDAY, THANK YOU, etc.) have any usable data?
print()
print("=== V1 TOP CLASSES CHECK ===")
v1_top = ['MONDAY', 'THANK YOU', 'MOTHER', 'PENCIL', 'CHRISTMAS', 'FRIDAY', 'TUESDAY', 'INDIA', 'ANGRY', 'NAME']
gmap = {g['gloss']: g for g in ranked}
for name in v1_top:
    if name in gmap:
        g = gmap[name]
        print(f"  {name:15s} total={g['total']:2d} both={g['both']:2d} one={g['one']:2d} none={g['none']:2d} usable={g['usable_count']:2d} usable%={g['usable_pct']:5.1f}%")
