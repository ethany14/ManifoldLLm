import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description='Convert an authorized dataset CSV to the bridge schema (no model inference).')
p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset',required=True)
p.add_argument('--person-id',required=True);p.add_argument('--text',required=True)
p.add_argument('--personality',default='');p.add_argument('--personality-source',default='questionnaire')
p.add_argument('--emotion',default='');p.add_argument('--emotion-kind',choices=['observed','predicted','missing'],default='missing')
p.add_argument('--language',default='');p.add_argument('--dialogue-id',default='');p.add_argument('--turn-index',default='');p.add_argument('--timestamp',default='')
a=p.parse_args()
fields=['dataset','person_id','dialogue_id','turn_index','timestamp','text','personality_json','personality_source','emotion_label','emotion_kind','vad_v','vad_a','vad_d','vad_kind','language']
with open(a.input,encoding='utf-8-sig',newline='') as f: data=list(csv.DictReader(f))
if not data: raise SystemExit('Input has no rows')
needed=[a.person_id,a.text]+[v for v in (a.personality,a.emotion,a.dialogue_id,a.turn_index,a.timestamp) if v]
missing=set(needed)-set(data[0]);
if missing: raise SystemExit(f'Input columns missing: {sorted(missing)}')
if a.emotion and a.emotion_kind=='missing': raise SystemExit('Specify --emotion-kind observed or predicted for populated emotion labels')
if not a.emotion and a.emotion_kind!='missing': raise SystemExit('Cannot mark missing emotion as observed/predicted')
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in data:
  pval=r[a.personality].strip() if a.personality else ''
  if pval:
   try: obj=json.loads(pval);assert isinstance(obj,dict)
   except (ValueError,AssertionError): raise SystemExit('personality column must contain a JSON object string')
  values=dict.fromkeys(fields,'');values.update(dataset=a.dataset,person_id=r[a.person_id],text=r[a.text],personality_json=pval,personality_source=a.personality_source if pval else 'missing',emotion_label=r[a.emotion] if a.emotion else '',emotion_kind=a.emotion_kind, vad_kind='missing',language=a.language)
  for dst,key in [('dialogue_id',a.dialogue_id),('turn_index',a.turn_index),('timestamp',a.timestamp)]:
   if key:values[dst]=r[key]
  w.writerow(values)
print(f'Wrote {len(data)} rows to {out}')
