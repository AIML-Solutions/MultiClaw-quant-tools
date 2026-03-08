#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib import request, parse

ALPACA_BASE = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("ALPACA_PAPER_KEY")
    or os.getenv("ALPACA_LIVE_KEY")
)
ALPACA_SECRET = (
    os.getenv("ALPACA_API_SECRET")
    or os.getenv("ALPACA_PAPER_SECRET")
    or os.getenv("ALPACA_LIVE_SECRET")
)
ALPACA_ACCOUNT_ID = os.getenv("ALPACA_ACCOUNT_ID", "alpaca-paper-default")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://quant:quant_dev_change_me@127.0.0.1:5432/quant")


def normalized_base(base: str) -> str:
    b = (base or "").strip().replace("export", "").strip()
    if b.endswith("/v2"):
        b = b[:-3]
    return b.rstrip("/")


def get(path, query=None):
    url = normalized_base(ALPACA_BASE) + path
    if query:
        url += "?" + parse.urlencode(query)
    req = request.Request(url)
    req.add_header("APCA-API-KEY-ID", ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Accept", "application/json")
    with request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise SystemExit("Missing ALPACA_API_KEY/ALPACA_API_SECRET")

    account = get("/v2/account")
    positions = get("/v2/positions")
    orders = get("/v2/orders", {"status": "all", "limit": 200, "nested": "false"})

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "account": {
            "status": account.get("status"),
            "base_currency": account.get("currency", "USD"),
            "equity": account.get("equity"),
            "raw": account,
        },
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "avg_entry_price": p.get("avg_entry_price"),
                "market_price": p.get("current_price"),
                "market_value": p.get("market_value"),
                "unrealized_pl": p.get("unrealized_pl"),
                "raw": p,
            }
            for p in positions
        ],
        "orders": [
            {
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "status": o.get("status"),
                "qty": o.get("qty"),
                "filled_qty": o.get("filled_qty"),
                "submitted_at": o.get("submitted_at"),
                "updated_at": o.get("updated_at"),
                "raw": o,
            }
            for o in orders
        ],
    }

    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            tmp = f.name

        ingest_py = os.path.join(os.path.dirname(__file__), "ingest_broker_snapshot.py")
        cmd = [
        "python3",
        ingest_py,
        "--provider",
        "alpaca",
        "--account-id",
        ALPACA_ACCOUNT_ID,
        "--mode",
        "paper",
        "--snapshot",
        tmp,
        "--db-url",
        DATABASE_URL,
        ]
        subprocess.run(cmd, check=True)
        print("Alpaca paper pull+ingest complete")


    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

if __name__ == "__main__":
    main()
