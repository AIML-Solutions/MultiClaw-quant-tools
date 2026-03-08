#!/usr/bin/env python3
import json, os, requests
from datetime import datetime, timezone
from pathlib import Path

BASE=os.getenv('IBEAM_GATEWAY_BASE_URL','https://localhost:5001').rstrip('/')
VERIFY=False
OUT=Path('/home/aimls-dtd/.openclaw/workspace/projects/quantconnect/services/ingestion/sample_ibkr_snapshot.json')

s=requests.Session()
status=s.get(f'{BASE}/v1/api/iserver/auth/status',verify=VERIFY,timeout=15)
if status.status_code!=200:
    raise SystemExit(f'IBKR auth status http={status.status_code}')
st=status.json()
if not st.get('authenticated'):
    raise SystemExit('IBKR not authenticated')

accts=s.get(f'{BASE}/v1/api/portfolio/accounts',verify=VERIFY,timeout=20).json()
if not accts:
    raise SystemExit('No IBKR accounts returned')
acct=accts[0].get('accountId') or accts[0].get('id')

pos=s.get(f'{BASE}/v1/api/portfolio/{acct}/positions/0',verify=VERIFY,timeout=25)
positions=pos.json() if pos.ok else []
orders=s.get(f'{BASE}/v1/api/iserver/account/orders',params={'force':'false'},verify=VERIFY,timeout=25)
orders_json=orders.json() if orders.ok else {}
order_list=orders_json.get('orders') or []

snapshot={
  'as_of': datetime.now(timezone.utc).isoformat(),
  'account': {'status':'AUTHENTICATED','base_currency':'USD','account_id':acct,'auth':st},
  'positions': positions if isinstance(positions,list) else [],
  'orders': order_list if isinstance(order_list,list) else []
}
OUT.write_text(json.dumps(snapshot,indent=2))
print(str(OUT))
print(f'account={acct} positions={len(snapshot["positions"])} orders={len(snapshot["orders"])}')

# ingest into quant db via existing normalizer
ingest='/home/aimls-dtd/.openclaw/workspace/projects/quantconnect/services/ingestion/ingest_broker_snapshot.py'
os.system(f'python3 {ingest} --provider ibkr --account-id {acct} --mode live --snapshot {OUT}')
