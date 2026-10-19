import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--category',default='');a=p.parse_args()
with (Path(__file__).resolve().parents[1]/'datasets/catalog.csv').open(encoding='utf8',newline='') as f:
 rows=list(csv.DictReader(f))
for r in rows:
 if not a.category or a.category.lower() in r['category'].lower(): print(f"{r['dataset']:34} {r['category']:26} {r['source_url']}")
print(f"Displayed {sum(not a.category or a.category.lower() in r['category'].lower() for r in rows)} / {len(rows)} records")
