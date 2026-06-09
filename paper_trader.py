#!/usr/bin/env python3
"""
Paper Trading Module for Event Contract Strategy.

Simulates event contract trades using live Binance data.
- Fetches latest klines from Binance REST API
- Generates signals on each completed bar
- Tracks paper positions through expiry
- Logs all trades to CSV

Usage:
    python paper_trader.py              # One check cycle
    python paper_trader.py --watch      # Watch mode (continuous)
    python paper_trader.py --report     # Show current stats
    python paper_trader.py --backfill   # Process recent history
"""

import sys, json, time, uuid
import signal as os_signal
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd
import ccxt

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.resolve()))
from data_loader import TimeframeData, resample_ohlcv
from production import ProductionSignalGenerator

PROJECT_DIR = Path(__file__).parent.resolve()
STATE_FILE = PROJECT_DIR / 'paper_state.json'
TRADE_LOG = PROJECT_DIR / 'paper_trades.csv'
SIGNAL_LOG = PROJECT_DIR / 'paper_signals.csv'
LOOKBACK = 500  # Bars of history for indicator warm-up


class LiveDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    
    def fetch_klines(self, tf: str, limit: int = None) -> pd.DataFrame:
        if limit is None:
            limit = LOOKBACK
        try:
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', tf, limit=limit)
        except Exception as e:
            print(f"  ⚠ Binance {tf}: {e}")
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_recent(self, lookback: int = None) -> Dict[str, TimeframeData]:
        if lookback is None:
            lookback = LOOKBACK
        result = {}
        for tf in ['3m', '5m', '30m', '1h']:
            df = self.fetch_klines(tf, limit=lookback)
            if not df.empty:
                result[tf] = TimeframeData(df=df, name=tf)
        if '5m' in result:
            result['10m'] = TimeframeData(df=resample_ohlcv(result['5m'].df, '10min'), name='10m')
        return result


class PaperTrader:
    def __init__(self):
        self.state = self._load()
        self.fetcher = LiveDataFetcher()
        self.generator = None
        self.data = None
    
    def _load(self) -> dict:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return {'positions': [], 'stats': {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}}
    
    def _save(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str))
    
    def _refresh(self):
        print("📡 Fetching Binance data...")
        self.data = self.fetcher.get_recent()
        for tf, td in self.data.items():
            print(f"  {tf}: {len(td.df)} bars → {td.df.index[-1]}")
        print("🔧 Computing signals...")
        self.generator = ProductionSignalGenerator(self.data)
    
    def _tz(self, ts) -> datetime:
        """Ensure timezone-aware datetime."""
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize('UTC') if ts.tz is None else ts
        dt = datetime.fromisoformat(str(ts))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    
    def check(self) -> list:
        """One check cycle: settle expired + generate new signals."""
        self._refresh()
        now = datetime.now(timezone.utc)
        new_trades = []
        
        # 1. Settle expired positions
        for pos in self.state['positions']:
            if pos['status'] != 'open':
                continue
            expiry = self._tz(pos['expiry_time'])
            if now < expiry:
                continue
            
            # Find exit price from data
            tf = pos['tf']
            if tf not in self.data:
                continue
            df = self.data[tf].df
            expiry_bars = df[df.index >= expiry]
            if expiry_bars.empty:
                continue
            
            exit_price = float(expiry_bars.iloc[0]['close'])
            pos['exit_price'] = exit_price
            pos['settled_at'] = now.isoformat()
            won = (exit_price > pos['entry_price']) if pos['direction'] == 'LONG' else (exit_price < pos['entry_price'])
            pos['status'] = 'won' if won else 'lost'
            pos['pnl_pct'] = 60.0 if won else -100.0
            
            self.state['stats']['total'] += 1
            if won: self.state['stats']['wins'] += 1
            else: self.state['stats']['losses'] += 1
            self.state['stats']['pnl'] += pos['pnl_pct']
            
            emoji = '✅' if won else '❌'
            print(f"  {emoji} SETTLED {tf} {pos['direction']}: {pos['entry_price']:.1f}→{exit_price:.1f} "
                  f"({'WON +60%' if won else 'LOST -100%'}) [{pos['rule_name']}]")
            self._log_trade(pos)
        
        # 2. Generate new signals
        for tf in ['30m', '10m', '3m']:
            if tf not in self.data:
                continue
            df = self.data[tf].df
            if len(df) < 2:
                continue
            
            latest_idx = df.index[-2]  # Last completed bar
            
            # Skip if already processed
            key = f'last_{tf}'
            if self.state.get(key) == str(latest_idx):
                continue
            
            signal = self.generator.generate(latest_idx, tf)
            self.state[key] = str(latest_idx)
            
            if not signal:
                continue
            
            # Check expiry is in future
            expiry = self._tz(signal.expires_at)
            if expiry <= now:
                continue
            
            # Cooldown: skip if active position in same TF
            if any(p['tf'] == tf and p['status'] == 'open' for p in self.state['positions']):
                continue
            
            minutes_left = (expiry - now).total_seconds() / 60
            print(f"  📊 NEW {tf} {signal.direction}: entry={signal.entry_price:.1f} "
                  f"expires_in={minutes_left:.0f}min rule={signal.rule_name}")
            
            pos = {
                'id': str(uuid.uuid4())[:8],
                'tf': tf, 'direction': signal.direction,
                'entry_time': str(signal.timestamp), 'entry_price': signal.entry_price,
                'expiry_time': str(signal.expires_at),
                'rule_name': signal.rule_name, 'confidence': signal.confidence,
                'status': 'open', 'exit_price': None, 'pnl_pct': None, 'settled_at': None,
            }
            self.state['positions'].append(pos)
            new_trades.append(pos)
        
        self._save()
        return new_trades
    
    def _log_trade(self, pos: dict):
        row = {k: pos[k] for k in ['id','tf','direction','entry_time','entry_price','expiry_time',
                                     'exit_price','rule_name','confidence','status','pnl_pct','settled_at']}
        hdr = not TRADE_LOG.exists()
        pd.DataFrame([row]).to_csv(TRADE_LOG, mode='a', header=hdr, index=False)
    
    def report(self) -> str:
        s = self.state['stats']
        wr = s['wins'] / max(1, s['total']) * 100
        open_pos = [p for p in self.state['positions'] if p['status'] == 'open']
        
        lines = [
            f"\n{'='*50}",
            f"  Paper Trading Report",
            f"{'='*50}",
            f"  Trades: {s['total']} | Wins: {s['wins']} | Losses: {s['losses']}",
            f"  Win Rate: {wr:.1f}% | Total PnL: {s['pnl']:+.1f}%",
            f"  Open Positions: {len(open_pos)}",
        ]
        if open_pos:
            lines.append("")
            for p in open_pos:
                exp = self._tz(p['expiry_time'])
                left = (exp - datetime.now(timezone.utc)).total_seconds() / 60
                lines.append(f"  🔄 {p['tf']} {p['direction']} @ {p['entry_price']:.1f} "
                           f"expires in {left:.0f}min [{p['rule_name']}]")
        return '\n'.join(lines)
    
    def watch(self, interval: int = 30):
        print(f"\n👁 Watch mode — checking every {interval}s (Ctrl+C to stop)\n")
        running = True
        os_signal.signal(os_signal.SIGINT, lambda *_: setattr(sys.modules[__name__], 'running', False))
        
        while running:
            try:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{ts}] Checking...")
                self.check()
                print(self.report())
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  ⚠ {e}")
                time.sleep(interval)
    
    def backfill(self, days: int = 3):
        """Process recent historical bars to catch up."""
        print(f"📜 Backfilling last {days} days...")
        self._refresh()
        now = datetime.now(timezone.utc)
        cutoff = now - pd.Timedelta(days=days)
        
        settled = 0
        for tf in ['30m', '10m', '3m']:
            if tf not in self.data:
                continue
            df = self.data[tf].df
            
            for i in range(len(df) - 1):
                idx = df.index[i]
                if idx.tz_localize('UTC') if idx.tz is None else idx < cutoff:
                    continue
                
                expiry = df.index[i + 1]
                if expiry.tz_localize('UTC') if expiry.tz is None else expiry > now:
                    continue
                
                signal = self.generator.generate(idx, tf)
                if not signal:
                    continue
                
                entry_price = float(df['close'].iloc[i])
                exit_price = float(df['close'].iloc[i + 1])
                won = (exit_price > entry_price) if signal.direction == 'LONG' else (exit_price < entry_price)
                
                pos = {
                    'id': str(uuid.uuid4())[:8],
                    'tf': tf, 'direction': signal.direction,
                    'entry_time': str(idx), 'entry_price': entry_price,
                    'expiry_time': str(expiry),
                    'rule_name': signal.rule_name, 'confidence': signal.confidence,
                    'status': 'won' if won else 'lost',
                    'exit_price': exit_price,
                    'pnl_pct': 60.0 if won else -100.0,
                    'settled_at': now.isoformat(),
                }
                self.state['positions'].append(pos)
                self.state['stats']['total'] += 1
                if won: self.state['stats']['wins'] += 1
                else: self.state['stats']['losses'] += 1
                self.state['stats']['pnl'] += pos['pnl_pct']
                self._log_trade(pos)
                settled += 1
        
        self._save()
        print(f"  Settled {settled} historical trades")
        print(self.report())


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--watch', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--backfill', type=int, default=0, help='Backfill N days')
    p.add_argument('--interval', type=int, default=30)
    args = p.parse_args()
    
    trader = PaperTrader()
    
    if args.report:
        print(trader.report())
    elif args.backfill:
        trader.backfill(args.backfill)
    elif args.watch:
        trader.watch(args.interval)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Check cycle...")
        trader.check()
        print(trader.report())


if __name__ == '__main__':
    main()
