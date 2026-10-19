import csv,sys,json
from pathlib import Path
if len(sys.argv)!=2: raise SystemExit('Usage: python scripts/validate_bridge.py file.csv')
with Path(sys.argv[1]).open(encoding='utf8',newline='') as f:r=list(csv.DictReader(f))
required={'dataset','person_id','text','personality_json','personality_source','emotion_label','emotion_kind','vad_v','vad_a','vad_d','vad_kind'}
if not r:raise SystemExit('No rows')
missing=required-set(r[0]);assert not missing, f'Missing fields: {missing}'
issues=[]
for i,x in enumerate(r,2):
 if not x['dataset'] or not x['person_id'] or not x['text']:issues.append(f'row {i}: dataset/person_id/text required')
 if x['emotion_kind'] not in ('observed','predicted','missing') or x['vad_kind'] not in ('observed','predicted','missing'):issues.append(f'row {i}: invalid provenance')
 if x['emotion_label'] and x['emotion_kind']=='missing':issues.append(f'row {i}: emotion label but missing provenance')
 if x['personality_json']:
  try:assert isinstance(json.loads(x['personality_json']),dict)
  except (ValueError,AssertionError):issues.append(f'row {i}: personality_json is not an object')
 if x['vad_kind']!='missing' and not all(x[k] for k in ('vad_v','vad_a','vad_d')):issues.append(f'row {i}: incomplete VAD')
 if x['vad_kind']=='missing' and any(x[k] for k in ('vad_v','vad_a','vad_d')):issues.append(f'row {i}: VAD value without provenance')
print('Rows:',len(r),'Issues:',len(issues));print('\n'.join(issues[:30]));sys.exit(bool(issues))
