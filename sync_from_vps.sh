#!/bin/bash
# Sync paper trading data from VPS and generate report
# Run every 4 hours via cron

set -e

SSH_KEY="$HOME/.ssh/openclaw_vps"
VPS="root@49.13.164.197"
VPS_DIR="/root/events-contract-strategy"
LOCAL_DIR="/mnt/c/Users/12645/events-contract-strategy"
SYNC_DIR="$LOCAL_DIR/vps_sync"
TIMESTAMP=$(date +%Y%m%d_%H%M)

mkdir -p "$SYNC_DIR"

echo "=== Paper Trading Sync — $(date '+%Y-%m-%d %H:%M') ==="

# 1. Pull latest data from VPS (ignore missing log)
echo "📡 Pulling from VPS..."
scp -i "$SSH_KEY" -q \
  "$VPS:$VPS_DIR/paper_trades.csv" \
  "$VPS:$VPS_DIR/paper_state.json" \
  "$SYNC_DIR/" 2>/dev/null
scp -i "$SSH_KEY" -q \
  "$VPS:$VPS_DIR/logs/paper_cron.log" \
  "$SYNC_DIR/" 2>/dev/null || true

# 2. Archive a dated copy
cp "$SYNC_DIR/paper_trades.csv" "$SYNC_DIR/paper_trades_$TIMESTAMP.csv" 2>/dev/null || true
cp "$SYNC_DIR/paper_state.json" "$SYNC_DIR/paper_state_$TIMESTAMP.json" 2>/dev/null || true

# 3. Generate report
echo ""
python3 -c "
import pandas as pd
from pathlib import Path

f = Path('$SYNC_DIR/paper_trades.csv')
if not f.exists():
    print('No trades yet.')
    exit()

df = pd.read_csv(f)
if len(df) == 0:
    print('No trades yet.')
    exit()

total = len(df)
wins = (df['status'] == 'won').sum()
losses = (df['status'] == 'lost').sum()
wr = wins / total * 100
pnl = df['pnl_pct'].sum()

print(f'Total: {total} | Wins: {wins} | Losses: {losses}')
print(f'Win Rate: {wr:.1f}% | Total PnL: {pnl:+.0f}%')
print()

# By timeframe
for tf in ['3m', '10m', '30m']:
    t = df[df['tf'] == tf]
    if len(t) == 0:
        continue
    w = (t['status'] == 'won').sum()
    twr = w / len(t) * 100
    print(f'{tf}: {len(t)} trades, WR={twr:.1f}%')

# Last 10 trades
print()
print('Last 10 trades:')
for _, r in df.tail(10).iterrows():
    emoji = '✅' if r['status'] == 'won' else '❌'
    print(f'  {emoji} {r[\"tf\"]} {r[\"direction\"]} {r[\"rule_name\"]} → {r[\"status\"].upper()} ({r[\"pnl_pct\"]:+.0f}%)')

# VPS errors (last 5 lines)
log = Path('$SYNC_DIR/paper_cron.log')
if log.exists():
    content = log.read_text()
    errors = [l for l in content.split('\n') if 'Error' in l or 'Traceback' in l]
    if errors:
        print()
        print('⚠ VPS errors (last 5):')
        for e in errors[-5:]:
            print(f'  {e[:120]}')
"

echo ""
echo "✅ Sync complete — $SYNC_DIR/"