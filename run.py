#!/usr/bin/env python3
"""
Event Contract Consensus Strategy — Main Entry Point.

Usage:
    python run.py                          # All timeframes
    python run.py --tf 30m                 # Single timeframe
    python run.py --tf 30m --threshold 20  # Custom consensus threshold
    python run.py --tf 30m --detailed      # Show trade list
"""

import argparse
import sys
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.resolve()))

from data_loader import load_btc_data
from engine import (
    ConsensusEngine, EventContractBacktester, 
    BacktestResult, format_result, print_trade_list,
    CONSENSUS_THRESHOLD
)
import pandas as pd
import numpy as np


TARGET_TIMEFRAMES = ['3m', '10m', '30m', '1h']


def run_backtest(tf: str, threshold: int = CONSENSUS_THRESHOLD,
                 cooldown: int = 0, verbose: bool = True) -> BacktestResult:
    """Run backtest for a single timeframe."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {tf} Event Contract Backtest (threshold ≥{threshold}/25)")
        print(f"{'='*60}")
    
    # Load data
    needed_tfs = [tf]
    if tf != '1h':
        needed_tfs.append('1h')  # For multi-timeframe context
    
    all_data = load_btc_data(needed_tfs)
    target_data = all_data[tf]
    higher_tf = {k: v for k, v in all_data.items() if k != tf}
    
    if verbose:
        print(f"  Data: {len(target_data.df):,} bars, "
              f"{target_data.df.index[0]} → {target_data.df.index[-1]}")
    
    # Build engine
    engine = ConsensusEngine(tf, target_data, higher_tf)
    
    if verbose:
        print("  Computing 25 systems...")
    
    signal_matrix = engine.compute_all_signals()
    
    if verbose:
        # Show consensus distribution
        consensus = engine.consensus.dropna()
        print(f"  Consensus range: [{consensus.min():+.0f}, {consensus.max():+.0f}]")
        
        # Count signals above threshold
        high_consensus = (consensus >= threshold).sum()
        high_consensus_short = (consensus <= -threshold).sum()
        print(f"  Bars ≥ |{threshold}|: {high_consensus + high_consensus_short} "
              f"(long={high_consensus}, short={high_consensus_short})")
        print(f"  That's {(high_consensus + high_consensus_short)/len(consensus)*100:.3f}% of bars")
        
        # System correlation summary
        if verbose:
            corr = signal_matrix.corr()
            # Average absolute correlation between systems
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            avg_corr = corr.values[mask].mean()
            print(f"  Avg system inter-correlation: {avg_corr:.3f}")
    
    # Run backtest
    backtester = EventContractBacktester(engine, threshold=threshold, cooldown_bars=cooldown)
    result = backtester.run()
    
    if verbose:
        print(f"\n{format_result(result)}")
    
    return result


def run_all(threshold: int = CONSENSUS_THRESHOLD, cooldown: int = 0):
    """Run all timeframes and produce summary."""
    
    results = {}
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Event Contract Consensus Strategy — Full Report   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Threshold: ≥{threshold}/25  |  Payoff: +60% / -100%")
    print(f"  Break-even WR: 62.5%")
    print()
    
    for tf in TARGET_TIMEFRAMES:
        try:
            result = run_backtest(tf, threshold=threshold, cooldown=cooldown)
            results[tf] = result
        except Exception as e:
            print(f"  ❌ {tf} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"{'SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'TF':<8} {'Trades':<8} {'Win Rate':<10} {'Total PnL':<12} "
          f"{'Avg PnL':<10} {'Expectancy':<12} {'L/W':<8}")
    print("-" * 80)
    
    for tf, r in results.items():
        print(f"{tf:<8} {len(r.trades):<8} {r.win_rate:<10.1f}% "
              f"{r.total_pnl_pct:<+11.1f}% {r.avg_pnl_pct:<+9.2f}% "
              f"{r.expectancy:<+11.2f}% {len([t for t in r.trades if t.pnl_pct>0])}/{len(r.trades)}")
    
    print("-" * 80)
    
    total_trades = sum(len(r.trades) for r in results.values())
    total_pnl = sum(r.total_pnl_pct for r in results.values())
    print(f"{'TOTAL':<8} {total_trades:<8} {'—':<10} {total_pnl:<+11.1f}% "
          f"{'—':<10} {'—':<12} {'—':<8}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Event Contract Consensus Strategy')
    parser.add_argument('--tf', type=str, default=None,
                        help='Single timeframe (3m, 10m, 30m, 1h)')
    parser.add_argument('--threshold', type=int, default=CONSENSUS_THRESHOLD,
                        help=f'Consensus threshold (default: {CONSENSUS_THRESHOLD})')
    parser.add_argument('--cooldown', type=int, default=0,
                        help='Cooldown bars between trades')
    parser.add_argument('--detailed', action='store_true',
                        help='Show trade list')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path')
    
    args = parser.parse_args()
    
    if args.tf:
        result = run_backtest(args.tf, threshold=args.threshold,
                             cooldown=args.cooldown, verbose=True)
        if args.detailed and result.trades:
            print(f"\n--- Trade List ({args.tf}) ---")
            print_trade_list(result.trades)
        
        if args.output:
            report = format_result(result)
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")
    else:
        results = run_all(threshold=args.threshold, cooldown=args.cooldown)


if __name__ == '__main__':
    main()
