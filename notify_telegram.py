#!/usr/bin/env python3
"""
Telegram notification module for paper trading summaries.
Sends 4-hour trade summary to @SisieMarketAssistant_bot.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests

PROJECT_DIR = Path(__file__).parent.resolve()

# Token — loaded from VPS .env
import os
TELEGRAM_BOT_TOKEN = ""
_env_path = Path(__file__).parent.parent / 'sisie-assistant' / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            k, _, v = line.partition('=')
            if k.strip() in ('TELEGRAM_BOT_TOKEN', 'SISIEVPS_BOT_TOKEN'):
                TELEGRAM_BOT_TOKEN = v.strip()
                break
if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = os.environ.get('SISIEVPS_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = "5801962948"


def load_trades() -> list:
    """Load all trades from CSV."""
    import pandas as pd
    f = PROJECT_DIR / 'paper_trades.csv'
    if not f.exists():
        return []
    df = pd.read_csv(f)
    return df.to_dict('records')


def get_recent_trades(hours: int = 4) -> list:
    """Get trades from the last N hours."""
    trades = load_trades()
    if not trades:
        return []
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = []
    for t in trades:
        settled = t.get('settled_at', '')
        if not settled or pd.isna(settled):
            continue
        try:
            settled_dt = datetime.fromisoformat(str(settled))
            if settled_dt.tzinfo is None:
                settled_dt = settled_dt.replace(tzinfo=timezone.utc)
            if settled_dt >= cutoff:
                recent.append(t)
        except:
            pass
    return recent


def get_overall_stats() -> dict:
    """Get all-time stats."""
    trades = load_trades()
    if not trades:
        return {'total': 0, 'wins': 0, 'losses': 0, 'wr': 0, 'pnl': 0}
    
    total = len(trades)
    wins = sum(1 for t in trades if t.get('status') == 'won')
    losses = sum(1 for t in trades if t.get('status') == 'lost')
    wr = wins / max(1, total) * 100
    pnl = sum(t.get('pnl_pct', 0) for t in trades)
    
    return {'total': total, 'wins': wins, 'losses': losses, 'wr': wr, 'pnl': pnl}


def format_summary(recent_trades: list, hours: int = 4) -> str:
    """Format a Telegram-friendly summary."""
    if not recent_trades:
        return f"📊 Paper Trading ({hours}h)\n\nNo trades settled in this period."
    
    total = len(recent_trades)
    wins = sum(1 for t in recent_trades if t.get('status') == 'won')
    losses = sum(1 for t in recent_trades if t.get('status') == 'lost')
    wr = wins / max(1, total) * 100
    pnl = sum(t.get('pnl_pct', 0) for t in recent_trades)
    
    # Break down by timeframe
    by_tf = {}
    for t in recent_trades:
        tf = t.get('tf', '?')
        if tf not in by_tf:
            by_tf[tf] = {'total': 0, 'wins': 0, 'pnl': 0}
        by_tf[tf]['total'] += 1
        if t.get('status') == 'won':
            by_tf[tf]['wins'] += 1
        by_tf[tf]['pnl'] += t.get('pnl_pct', 0)
    
    # Overall stats
    overall = get_overall_stats()
    
    lines = [
        f"📊 事件合约 Paper Trading",
        f"━━━━━━━━━━━━━━━━━━",
        f"",
        f"⏱ 过去 {hours} 小时",
        f"  交易: {total} | 胜: {wins} | 负: {losses}",
        f"  胜率: {wr:.1f}% | PnL: {pnl:+.0f}%",
        f"",
    ]
    
    # Per timeframe
    for tf in ['3m', '10m', '30m']:
        if tf in by_tf:
            s = by_tf[tf]
            twr = s['wins'] / max(1, s['total']) * 100
            lines.append(f"  {tf}: {s['total']}笔, WR={twr:.0f}%, {s['pnl']:+.0f}%")
    
    lines.extend([
        f"",
        f"📈 累计统计",
        f"  总交易: {overall['total']} | 胜率: {overall['wr']:.1f}%",
        f"  累计 PnL: {overall['pnl']:+.0f}%",
        f"",
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
    ])
    
    return '\n'.join(lines)


def send_telegram(message: str, token: str = None, chat_id: str = None):
    """Send message via Telegram bot."""
    token = token or TELEGRAM_BOT_TOKEN
    chat_id = chat_id or TELEGRAM_CHAT_ID
    
    if 'YOUR_BOT' in token:
        print("⚠ Telegram token not configured")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    try:
        resp = requests.post(url, json={
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
        }, timeout=10)
        
        if resp.status_code == 200:
            print("✅ Telegram sent")
            return True
        else:
            print(f"⚠ Telegram error: {resp.status_code} {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"⚠ Telegram error: {e}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=int, default=4, help='Summary period in hours')
    p.add_argument('--token', type=str, default=None, help='Bot token override')
    p.add_argument('--chat-id', type=str, default=None, help='Chat ID override')
    p.add_argument('--dry-run', action='store_true', help='Print only, no send')
    args = p.parse_args()
    
    recent = get_recent_trades(args.hours)
    msg = format_summary(recent, args.hours)
    
    if args.dry_run:
        print(msg)
    else:
        print(msg)
        print()
        send_telegram(msg, args.token, args.chat_id)


if __name__ == '__main__':
    import pandas as pd
    main()
