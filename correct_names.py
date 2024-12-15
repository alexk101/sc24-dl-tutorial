from pathlib import Path
from collections import defaultdict
import os

target = Path('/home/alexk101/sc24-dl-tutorial/logs/mp/4MP')
data = defaultdict(list)

for x in target.iterdir():
    name = x.name
    if 'L' in name:
        data['Layers'].append(x)
    elif 'emb' in name and 'val' not in name:
        data['Embedding'].append(x)
    elif 'val' in name:
        data['Data'].append(x)

for key, val in data.items():
    print(key)
    for x in val:
        print(x)