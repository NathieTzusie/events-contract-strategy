"""
Consensus Engine + Event Contract Backtester.

Core logic:
1. 25 systems each produce +1/-1/0 signals per bar
2. Consensus score = sum of all signals
3. Trade when |consensus| >= threshold (e.g., 22)
4. Direction = sign of consensus
5. PnL determined by comparing close[entry_bar + N] vs close[entry_bar]
   where N = expiry_bars for the target timeframe
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from data_loader import TimeframeData
import systems


# ═══════════════════════════════════════════════════════════════
# System Registry
# ═══════════════════════════════════════════════════════════════

# All 25 systems: (name, function, needs_higher_tf)
ALL_SYSTEMS = [
    # Group A: Short-term Momentum
    ("A1_RSI14", systems.rsi_bias, False),
    ("A2_Stoch", systems.stochastic_bias, False),
    ("A3_ROC5", systems.roc_bias, False),
    ("A4_CloseMom", systems.close_momentum, False),
    
    # Group B: Trend Structure
    ("B1_EMA5_20", systems.ema_cross_5_20, False),
    ("B2_EMA10_50", systems.ema_cross_10_50, False),
    ("B3_MACD", systems.macd_histogram, False),
    ("B4_SuperTrend", systems.supertrend, False),
    
    # Group C: Multi-Timeframe (needs 1H data)
    ("C1_TF_Trend", systems.higher_tf_trend, True),
    ("C2_TF_Momentum", systems.higher_tf_momentum, True),
    ("C3_TF_RSI", systems.higher_tf_rsi, True),
    ("C4_TF_ADX", systems.higher_tf_adx, True),
    
    # Group D: Volume
    ("D1_OBV", systems.obv_trend, False),
    ("D2_CloseLoc", systems.close_location, False),
    ("D3_CMF", systems.cmf_bias, False),
    
    # Group E: Volatility & Bands
    ("E1_BB_Pos", systems.bb_position, False),
    ("E2_BB_pctB", systems.bb_percent_b, False),
    ("E3_Keltner", systems.keltner_position, False),
    
    # Group F: Statistical
    ("F1_LinSlope", systems.linear_slope, False),
    ("F2_ZScore", systems.returns_zscore, False),
    ("F3_TrendConsistency", systems.trend_consistency, False),
    
    # Group G: Price Structure
    ("G1_HigherHighs", systems.higher_highs, False),
    ("G2_Donchian", systems.donchian_position, False),
    ("G3_DM", systems.directional_movement, False),
    
    # Group H: Alternative
    ("H1_Ichimoku", systems.ichimoku_bias, False),
    ("H2_SMA20_100", systems.sma_cross_20_100, False),
]

assert len(ALL_SYSTEMS) == 26, f"Expected 26 systems, got {len(ALL_SYSTEMS)}"


# ═══════════════════════════════════════════════════════════════
# Contract parameters
# ═══════════════════════════════════════════════════════════════

# Minutes per bar for each target timeframe
TF_MINUTES = {'3m': 3, '10m': 10, '30m': 30, '1h': 60}

# Payoff: win = +60%, loss = -100%
PAYOFF_RATIO = 0.6

# Consensus threshold: ≥22/25 systems must agree
CONSENSUS_THRESHOLD = 22


# ═══════════════════════════════════════════════════════════════
# Results data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    pnl_pct: float
    consensus_score: int
    tf: str


@dataclass
class BacktestResult:
    tf: str
    trades: List[Trade] = field(default_factory=list)
    total_signals: int = 0
    
    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct > 0)
    
    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct < 0)
    
    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.win_count / len(self.trades) * 100
    
    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades)
    
    @property
    def avg_pnl_pct(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl_pct / len(self.trades)
    
    @property
    def expectancy(self) -> float:
        """Expected return per trade in %"""
        if not self.trades:
            return 0.0
        win_r = self.win_rate / 100
        lose_r = 1 - win_r
        return win_r * PAYOFF_RATIO * 100 - lose_r * 100


# ═══════════════════════════════════════════════════════════════
# Consensus Engine
# ═══════════════════════════════════════════════════════════════

class ConsensusEngine:
    """Runs 25 systems and computes consensus signal per bar."""
    
    def __init__(self, target_tf: str, target_data: TimeframeData,
                 higher_tf_data: Optional[Dict[str, TimeframeData]] = None):
        self.target_tf = target_tf
        self.df = target_data.df
        self.higher_tf_data = higher_tf_data or {}
        
        # Primary higher TF for context (1H)
        self.df_1h = self.higher_tf_data.get('1h', None)
        
        # Results
        self.signals: Dict[str, pd.Series] = {}
        self.consensus: Optional[pd.Series] = None
    
    def compute_all_signals(self) -> pd.DataFrame:
        """Run all 25 systems, return signal matrix (rows=bars, cols=systems)."""
        signal_matrix = pd.DataFrame(index=self.df.index)
        
        for name, func, needs_htf in ALL_SYSTEMS:
            try:
                if needs_htf:
                    if self.df_1h is None:
                        # Skip HTF systems if 1H data not available
                        signal_matrix[name] = 0
                        continue
                    sig = func(self.df_1h.df, self.df)
                else:
                    sig = func(self.df)
                
                self.signals[name] = sig
                signal_matrix[name] = sig
            except Exception as e:
                print(f"  ⚠ {name} failed: {e}")
                signal_matrix[name] = 0
        
        # Compute consensus
        self.consensus = signal_matrix.sum(axis=1)
        
        return signal_matrix
    
    def get_trade_signals(self, threshold: int = CONSENSUS_THRESHOLD) -> pd.Series:
        """Return filtered signals: +1 (long), -1 (short), 0 (no trade)."""
        if self.consensus is None:
            self.compute_all_signals()
        
        trade_signal = pd.Series(0, index=self.df.index)
        trade_signal[self.consensus >= threshold] = 1
        trade_signal[self.consensus <= -threshold] = -1
        
        return trade_signal


# ═══════════════════════════════════════════════════════════════
# Event Contract Backtester
# ═══════════════════════════════════════════════════════════════

class EventContractBacktester:
    """
    Backtest event contracts: enter at bar close, exit after expiry_bars.
    
    No stop-loss, no take-profit. Pure binary outcome.
    PnL = +60% if direction correct, -100% if wrong.
    """
    
    def __init__(self, engine: ConsensusEngine, 
                 threshold: int = CONSENSUS_THRESHOLD,
                 cooldown_bars: int = 0):
        self.engine = engine
        self.threshold = threshold
        self.cooldown_bars = cooldown_bars
        self.tf = engine.target_tf
        self.df = engine.df
        self.expiry_bars = 1  # One bar = one contract period (3m → 3m expiry, etc.)
    
    def run(self) -> BacktestResult:
        """Run backtest."""
        # Compute all signals
        trade_signals = self.engine.get_trade_signals(self.threshold)
        
        result = BacktestResult(tf=self.tf)
        result.total_signals = int((trade_signals != 0).sum())
        
        cooldown_remaining = 0
        
        for i in range(len(self.df)):
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue
            
            sig = trade_signals.iloc[i]
            if sig == 0:
                continue
            
            # Check if we have enough future data
            exit_idx = i + self.expiry_bars
            if exit_idx >= len(self.df):
                continue
            
            entry_time = self.df.index[i]
            exit_time = self.df.index[exit_idx]
            entry_price = self.df['close'].iloc[i]
            exit_price = self.df['close'].iloc[exit_idx]
            direction = 'LONG' if sig == 1 else 'SHORT'
            consensus_score = int(self.engine.consensus.iloc[i])
            
            # Determine win/loss
            if direction == 'LONG':
                won = exit_price > entry_price
            else:
                won = exit_price < entry_price
            
            pnl_pct = PAYOFF_RATIO * 100 if won else -100.0
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                consensus_score=consensus_score,
                tf=self.tf,
            )
            result.trades.append(trade)
            
            # Cooldown
            cooldown_remaining = self.cooldown_bars
        
        return result


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════

def format_result(result: BacktestResult) -> str:
    """Format backtest result as markdown report."""
    trades = result.trades
    if not trades:
        return f"## {result.tf} — No trades found"
    
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct < 0]
    longs = [t for t in trades if t.direction == 'LONG']
    shorts = [t for t in trades if t.direction == 'SHORT']
    
    lines = [
        f"## {result.tf} Event Contract Backtest",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Signals | {result.total_signals} |",
        f"| Trades Executed | {len(trades)} |",
        f"| Win Rate | {result.win_rate:.1f}% |",
        f"| Wins / Losses | {len(wins)} / {len(losses)} |",
        f"| Total PnL | {result.total_pnl_pct:+.1f}% |",
        f"| Avg PnL/Trade | {result.avg_pnl_pct:+.2f}% |",
        f"| Expectancy/Trade | {result.expectancy:+.2f}% |",
        f"| Longs / Shorts | {len(longs)} / {len(shorts)} |",
    ]
    
    if longs:
        long_wr = sum(1 for t in longs if t.pnl_pct > 0) / len(longs) * 100
        lines.append(f"| Long Win Rate | {long_wr:.1f}% |")
    if shorts:
        short_wr = sum(1 for t in shorts if t.pnl_pct > 0) / len(shorts) * 100
        lines.append(f"| Short Win Rate | {short_wr:.1f}% |")
    
    avg_consensus = np.mean([abs(t.consensus_score) for t in trades])
    lines.append(f"| Avg |Consensus| | {avg_consensus:.1f}/25 |")
    
    # Date range
    lines.extend([
        f"",
        f"Period: {trades[0].entry_time} → {trades[-1].entry_time}",
        f"",
    ])
    
    return "\n".join(lines)


def print_trade_list(trades: List[Trade], max_show: int = 30):
    """Print trade list."""
    print(f"{'Entry':<20} {'Dir':<6} {'Entry$':<12} {'Exit$':<12} {'PnL%':<8} {'Cons':>5}")
    print("-" * 70)
    for t in trades[:max_show]:
        print(f"{str(t.entry_time):<20} {t.direction:<6} "
              f"{t.entry_price:<12.1f} {t.exit_price:<12.1f} "
              f"{t.pnl_pct:<+8.1f} {t.consensus_score:>+4d}")
    if len(trades) > max_show:
        print(f"  ... and {len(trades) - max_show} more trades")
