import csv
from pathlib import Path

freq_path = Path('e:/gessture/cislr_data/label_frequency.csv')
vocab_path = Path('e:/gessture/cislr_data/landmarks/recommended_v1.txt')

freq = {}
with open(freq_path, 'r', encoding='utf-8') as f:
    r = csv.reader(f)
    next(r)
    for g, c in r:
        freq[g] = int(c)

rec = []
with open(vocab_path, 'r', encoding='utf-8') as f:
    for l in f:
        rec.append(l.strip())

count_sum = sum(freq.get(g, 0) for g in rec)
print('selected_vocab_count=', len(rec))
print('sample_count_sum=', count_sum)
