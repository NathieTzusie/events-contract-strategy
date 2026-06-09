#!/usr/bin/env python3
"""
Walk-forward backtest for event contract strategy.

Simulates realistic live trading with monthly retraining.
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from sklearn.ensemble import GradientBoostingClassifier

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.resolve()))
from data_loader import load_btc_data
from engine import ConsensusEngine

PAYOFF = 0.6
POSITION_SIZE = 0.05
CONFIDENCE_THRESHOLD = 0.65
MIN_TRAIN_BARS = 5000


@dataclass
class WalkForwardResult:
    tf: str
    approach: str
    trades: List[dict] = field(default_factory=list)
    monthly_equity: List[dict] = field(default_factory=list)
    
    @property
    def total_trades(self): return len(self.trades)
    
    @property
    def win_rate(self):
        if not self.trades: return 0
        return sum(1 for t in self.trades if t['won']) / len(self.trades) * 100
    
    @property
    def total_pnl(self):
        if not self.trades: return 0
        eq = 1.0
        for t in self.trades:
            eq *= (1 + PAYOFF * POSITION_SIZE) if t['won'] else (1 - POSITION_SIZE)
        return (eq - 1) * 100
    
    @property
    def avg_monthly_return(self):
        if not self.monthly_equity: return 0
        return np.mean([m['return_pct'] for m in self.monthly_equity])


def rule_signal(row: pd.Series) -> int:
    """Rule-based strategy: contrarian with ADX/SMA confirmation."""
    hh = row.get('G1_HigherHighs', 0)
    zs = row.get('F2_ZScore', 0)
    adx_tf = row.get('C4_TF_ADX', 0)
    sma = row.get('H2_SMA20_100', 0)
    
    # Primary: contrarian + ADX
    if hh < 0 and zs < 0 and adx_tf > 0:
        return 1
    if hh > 0 and zs > 0 and adx_tf < 0:
        return -1
    # Secondary: trend pullback
    if sma > 0 and hh < 0 and zs < 0:
        return 1
    if sma < 0 and hh > 0 and zs > 0:
        return -1
    return 0


def walk_forward(tf: str, use_rules: bool = False,
                 conf_threshold: float = CONFIDENCE_THRESHOLD) -> WalkForwardResult:
    """Walk-forward backtest with monthly retraining."""
    
    approach = 'Rules' if use_rules else 'ML'
    print(f"\n{'='*60}")
    print(f"  Walk-Forward: {tf} ({approach})")
    print(f"{'='*60}")
    
    # Load data
    needed = [tf]
    if tf != '1h':
        needed.append('1h')
    all_data = load_btc_data(needed)
    
    engine = ConsensusEngine(tf, all_data[tf], {k: v for k, v in all_data.items() if k != tf})
    X = engine.compute_all_signals()
    df = engine.df
    y = (df['close'].shift(-1) > df['close']).astype(int)
    
    valid_mask = X.notna().all(axis=1) & y.notna()
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    result = WalkForwardResult(tf=tf, approach=approach)
    
    # Get monthly boundaries
    monthly_periods = pd.period_range(X_valid.index[0], X_valid.index[-1], freq='M')
    
    if len(monthly_periods) < 6:
        print(f"  Too few months: {len(monthly_periods)}")
        return result
    
    print(f"  Data: {len(X_valid):,} bars, {len(monthly_periods)} months")
    print(f"  First trade month: {monthly_periods[3]}")
    
    model = None
    
    for month_idx, period in enumerate(monthly_periods):
        if month_idx < 3:
            continue  # Warm-up
        
        # Test window: this month
        test_start = period.start_time
        test_end = period.end_time
        test_mask = (X_valid.index >= test_start) & (X_valid.index <= test_end)
        test_indices = X_valid.index[test_mask]
        
        if len(test_indices) < 10:
            continue
        
        # Train window: all data before this month
        train_mask = X_valid.index < test_start
        train_indices = X_valid.index[train_mask]
        
        if len(train_indices) < MIN_TRAIN_BARS:
            continue
        
        # Retrain model
        if not use_rules:
            X_train = X_valid.loc[train_indices]
            y_train = y_valid.loc[train_indices]
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, subsample=0.5
            )
            model.fit(X_train, y_train)
        
        # Test
        X_test = X_valid.loc[test_indices]
        y_test = y_valid.loc[test_indices]
        
        month_trades = []
        month_won = 0
        
        for i, (idx, row) in enumerate(X_test.iterrows()):
            if use_rules:
                sig = rule_signal(row)
                confidence = 0.80 if sig != 0 else 0
            else:
                proba = model.predict_proba(row.values.reshape(1, -1))[0]
                confidence = proba.max()
                sig = 1 if proba[1] > proba[0] else -1
            
            if confidence < conf_threshold:
                continue
            
            # Get actual outcome
            actual_up = bool(y_test.iloc[i])
            won = actual_up if sig == 1 else (not actual_up)
            
            # Entry/exit prices
            bar_pos = df.index.get_loc(idx)
            if bar_pos + 1 >= len(df):
                continue
            
            entry_price = df['close'].iloc[bar_pos]
            exit_price = df['close'].iloc[bar_pos + 1]
            
            trade = {
                'time': idx,
                'direction': 'LONG' if sig == 1 else 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'won': won,
                'confidence': confidence,
            }
            month_trades.append(trade)
            if won:
                month_won += 1
        
        # Monthly PnL
        if month_trades:
            eq = 1.0
            for t in month_trades:
                eq *= (1 + PAYOFF * POSITION_SIZE) if t['won'] else (1 - POSITION_SIZE)
            
            month_wr = month_won / len(month_trades) * 100
            month_ret = (eq - 1) * 100
            
            result.trades.extend(month_trades)
            result.monthly_equity.append({
                'month': str(period),
                'trades': len(month_trades),
                'win_rate': month_wr,
                'return_pct': month_ret,
            })
            
            bar = '█' * max(1, int(abs(month_ret) * 2))
            sign = '+' if month_ret > 0 else ''
            print(f"  [{period}] {len(month_trades):>3} trades, WR={month_wr:.0f}%, "
                  f"R={sign}{month_ret:.1f}% {bar}")
    
    # Summary
    if result.trades:
        print(f"\n  {'─'*50}")
        print(f"  TOTAL: {result.total_trades} trades, WR={result.win_rate:.1f}%, "
              f"Return={result.total_pnl:+.1f}%")
        print(f"  Avg Monthly: {result.avg_monthly_return:+.1f}%, "
              f"Profitable: {sum(1 for m in result.monthly_equity if m['return_pct']>0)}/"
              f"{len(result.monthly_equity)}")
    
    return result


if __name__ == '__main__':
    results = []
    
    for tf in ['30m', '10m', '3m', '1h']:
        for use_rules in [False, True]:
            try:
                r = walk_forward(tf, use_rules=use_rules)
                results.append(r)
            except Exception as e:
                print(f"  ❌ {tf} {'Rules' if use_rules else 'ML'}: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"{'FINAL WALK-FORWARD SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'TF':<7} {'Method':<7} {'Trades':<8} {'WR':<8} {'Total R':<10} "
          f"{'Avg Mo':<10} {'Prof Mo':<10}")
    print('-' * 80)
    
    for r in sorted(results, key=lambda x: (x.tf, x.approach)):
        if r.trades:
            prof_mo = f"{sum(1 for m in r.monthly_equity if m['return_pct']>0)}/{len(r.monthly_equity)}"
            print(f"{r.tf:<7} {r.approach:<7} {r.total_trades:<8} {r.win_rate:.1f}%     "
                  f"{r.total_pnl:+.1f}%      {r.avg_monthly_return:+.1f}%      {prof_mo}")
