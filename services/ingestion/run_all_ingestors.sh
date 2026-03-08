#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aimls-dtd/.openclaw/workspace/projects/quantconnect"
cd "$ROOT/infra"
set -a; source .env; set +a
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}"

cd "$ROOT/services/ingestion"

ALPACA_KEY_EFFECTIVE="${ALPACA_API_KEY:-${ALPACA_PAPER_KEY:-${ALPACA_LIVE_KEY:-}}}"
ALPACA_SECRET_EFFECTIVE="${ALPACA_API_SECRET:-${ALPACA_PAPER_SECRET:-${ALPACA_LIVE_SECRET:-}}}"
TRADIER_TOKEN_EFFECTIVE="${TRADIER_API_TOKEN:-${TRADIER_SANDBOX_TOKEN:-${TRADIER_LIVE_TOKEN:-}}}"

if [[ -n "$ALPACA_KEY_EFFECTIVE" && -n "$ALPACA_SECRET_EFFECTIVE" ]]; then
  echo "Running Alpaca ingestor..."
  python3 pull_alpaca_paper.py || echo "Alpaca ingestor failed"
else
  echo "Skipping Alpaca ingestor (missing key/secret vars)"
fi

if [[ -n "$TRADIER_TOKEN_EFFECTIVE" ]]; then
  echo "Running Tradier ingestor..."
  python3 pull_tradier_paper.py || echo "Tradier ingestor failed"
else
  echo "Skipping Tradier ingestor (missing token var)"
fi

if curl -sk https://localhost:5001/v1/api/iserver/auth/status | grep -q '"authenticated":true'; then
  echo "Running IBKR ingestor..."
  python3 pull_ibkr_gateway.py || echo "IBKR ingestor failed"
else
  echo "Skipping IBKR ingestor (gateway not authenticated)"
fi

echo "Ingestor cycle complete"
