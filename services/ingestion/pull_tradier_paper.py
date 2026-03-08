#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib import request

TRADIER_BASE = (
    os.getenv("TRADIER_PAPER_BASE_URL")
    or os.getenv("TRADIER_SANDBOX_BASE_URL")
    or os.getenv("TRADIER_LIVE_BASE_URL")
    or "https://sandbox.tradier.com/v1"
)
TRADIER_TOKEN = (
    os.getenv("TRADIER_API_TOKEN")
    or os.getenv("TRADIER_SANDBOX_TOKEN")
    or os.getenv("TRADIER_LIVE_TOKEN")
)
TRADIER_ACCOUNT_ID = (
    os.getenv("TRADIER_ACCOUNT_ID")
    or os.getenv("TRADIER_SANDBOX_ACCOUNT_ID")
    or os.getenv("TRADIER_LIVE_ACCOUNT_ID")
)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://quant:quant_dev_change_me@127.0.0.1:5432/quant")


def get(path):
    url = TRADIER_BASE.rstrip("/") + path
    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {TRADIER_TOKEN}")
    req.add_header("Accept", "application/json")
    with request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_account_id():
    global TRADIER_ACCOUNT_ID
    if TRADIER_ACCOUNT_ID:
        return
    profile = get("/user/profile")
    accounts = profile.get("profile", {}).get("account", [])
    if isinstance(accounts, dict):
        TRADIER_ACCOUNT_ID = accounts.get("account_number")
    elif isinstance(accounts, list) and accounts:
        TRADIER_ACCOUNT_ID = accounts[0].get("account_number")


def normalize_positions(data):
    if not isinstance(data, dict):
        return []
    positions_obj = data.get("positions", {})
    if not isinstance(positions_obj, dict):
        return []
    raw = positions_obj.get("position")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for p in raw:
        out.append({
            "symbol": p.get("symbol"),
            "qty": p.get("quantity"),
            "avg_entry_price": p.get("cost_basis"),
            "market_price": p.get("last"),
            "market_value": p.get("market_value"),
            "unrealized_pl": p.get("unrealized_gain_loss"),
            "raw": p,
        })
    return out


def normalize_orders(data):
    if not isinstance(data, dict):
        return []
    orders_obj = data.get("orders", {})
    if not isinstance(orders_obj, dict):
        return []
    raw = orders_obj.get("order")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for o in raw:
        out.append({
            "id": o.get("id"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "type": o.get("type"),
            "status": o.get("status"),
            "qty": o.get("quantity"),
            "filled_qty": o.get("exec_quantity"),
            "submitted_at": o.get("create_date"),
            "updated_at": o.get("transaction_date"),
            "raw": o,
        })
    return out


def main():
    if not TRADIER_TOKEN:
        raise SystemExit("Missing TRADIER_API_TOKEN")
    ensure_account_id()
    if not TRADIER_ACCOUNT_ID:
        raise SystemExit("Missing TRADIER_ACCOUNT_ID and could not discover from profile")

    balances = get(f"/accounts/{TRADIER_ACCOUNT_ID}/balances")
    positions = get(f"/accounts/{TRADIER_ACCOUNT_ID}/positions")
    orders = get(f"/accounts/{TRADIER_ACCOUNT_ID}/orders")

    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "account": {
            "status": "ACTIVE",
            "base_currency": "USD",
            "equity": balances.get("balances", {}).get("total_equity"),
            "raw": balances,
        },
        "positions": normalize_positions(positions),
        "orders": normalize_orders(orders),
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
        "tradier",
        "--account-id",
        TRADIER_ACCOUNT_ID,
        "--mode",
        "paper",
        "--snapshot",
        tmp,
        "--db-url",
        DATABASE_URL,
        ]
        subprocess.run(cmd, check=True)
        print("Tradier paper pull+ingest complete")


    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

if __name__ == "__main__":
    main()
