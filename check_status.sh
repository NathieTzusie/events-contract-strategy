#!/bin/bash
# Quick status check
cd /mnt/c/Users/12645/events-contract-strategy
echo "=== Paper Trading Status ==="
python3 -c "
from paper_trader import PaperTrader
t = PaperTrader()
print(t.report())
"
echo ""
echo "=== Recent Trades (last 5) ==="
python3 -c "
import pandas as pd
from pathlib import Path
f = Path('paper_trades.csv')
if f.exists():
    df = pd.read_csv(f)
    if len(df) > 0:
        for _, r in df.tail(5).iterrows():
            emoji = '✅' if r['status']=='won' else '❌'
            print(f'{emoji} {r[\"tf\"]} {r[\"direction\"]} @{r[\"entry_price\"]:.0f} {r[\"rule_name\"]} → {r[\"status\"].upper()} ({r[\"pnl_pct\"]:+.0f}%)')
"
