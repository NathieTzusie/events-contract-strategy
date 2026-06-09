"""
25 Independent Directional Systems for Event Contract Consensus Strategy.

Each system takes OHLCV data and returns a Series of signals:
  +1 = bullish
  -1 = bearish
   0 = neutral / no signal

Design principles:
- Minimize correlation between systems
- Each system should have standalone accuracy > 50%
- Systems operate on the target timeframe unless specified
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════
# Group A: Short-term Momentum (4 systems)
# ═══════════════════════════════════════════════════════════════

def rsi_bias(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI(14) > 50 → bullish, < 50 → bearish"""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=df.index)
    signal[rsi > 50] = 1
    signal[rsi < 50] = -1
    return signal


def stochastic_bias(df: pd.DataFrame, k_period: int = 14, 
                     d_period: int = 3) -> pd.Series:
    """Stochastic %K > %D → bullish, else bearish"""
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    pct_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    pct_d = pct_k.rolling(d_period).mean()
    signal = pd.Series(0, index=df.index)
    signal[pct_k > pct_d] = 1
    signal[pct_k < pct_d] = -1
    return signal


def roc_bias(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Rate of Change > 0 → bullish"""
    roc = df['close'].pct_change(period) * 100
    signal = pd.Series(0, index=df.index)
    signal[roc > 0] = 1
    signal[roc < 0] = -1
    return signal


def close_momentum(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """Close > Close[lookback] → bullish"""
    signal = pd.Series(0, index=df.index)
    signal[df['close'] > df['close'].shift(lookback)] = 1
    signal[df['close'] < df['close'].shift(lookback)] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Group B: Trend Structure (4 systems)
# ═══════════════════════════════════════════════════════════════

def ema_cross_5_20(df: pd.DataFrame) -> pd.Series:
    """EMA(5) > EMA(20) → bullish"""
    ema5 = df['close'].ewm(span=5, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[ema5 > ema20] = 1
    signal[ema5 < ema20] = -1
    return signal


def ema_cross_10_50(df: pd.DataFrame) -> pd.Series:
    """EMA(10) > EMA(50) → bullish"""
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[ema10 > ema50] = 1
    signal[ema10 < ema50] = -1
    return signal


def macd_histogram(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                   signal_p: int = 9) -> pd.Series:
    """MACD histogram > 0 → bullish"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
    histogram = macd_line - signal_line
    s = pd.Series(0, index=df.index)
    s[histogram > 0] = 1
    s[histogram < 0] = -1
    return s


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """SuperTrend: price above upper band → bullish, below lower → bearish"""
    atr = compute_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    trend = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        
        # Adjust bands
        if trend.iloc[i] == 1 and lower.iloc[i] < lower.iloc[i-1]:
            lower.iloc[i] = lower.iloc[i-1]
        if trend.iloc[i] == -1 and upper.iloc[i] > upper.iloc[i-1]:
            upper.iloc[i] = upper.iloc[i-1]
    
    return trend


# ═══════════════════════════════════════════════════════════════
# Group C: Multi-Timeframe Context (4 systems)
# ═══════════════════════════════════════════════════════════════

def higher_tf_trend(df_higher: pd.DataFrame, target_df: pd.DataFrame,
                    ema_period: int = 20) -> pd.Series:
    """Higher TF: Price > EMA → bullish (forward-filled to target TF).
    Uses PREVIOUS completed higher TF bar to avoid look-ahead."""
    # Shift to use PREVIOUS completed bar
    prev_close = df_higher['close'].shift(1)
    ema = prev_close.ewm(span=ema_period, adjust=False).mean()
    higher_signal = pd.Series(0, index=df_higher.index)
    higher_signal[prev_close > ema] = 1
    higher_signal[prev_close < ema] = -1
    return higher_signal.reindex(target_df.index, method='ffill').fillna(0)


def higher_tf_momentum(df_higher: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """Higher TF: EMA(5) > EMA(20) → bullish (uses previous completed bar)"""
    prev_close = df_higher['close'].shift(1)
    ema5 = prev_close.ewm(span=5, adjust=False).mean()
    ema20 = prev_close.ewm(span=20, adjust=False).mean()
    higher_signal = pd.Series(0, index=df_higher.index)
    higher_signal[ema5 > ema20] = 1
    higher_signal[ema5 < ema20] = -1
    return higher_signal.reindex(target_df.index, method='ffill').fillna(0)


def higher_tf_rsi(df_higher: pd.DataFrame, target_df: pd.DataFrame,
                  period: int = 14) -> pd.Series:
    """Higher TF: RSI(14) > 50 → bullish (uses previous completed bar)"""
    shifted = df_higher.copy()
    shifted['close'] = df_higher['close'].shift(1)
    higher_signal = rsi_bias(shifted, period)
    return higher_signal.reindex(target_df.index, method='ffill').fillna(0)


def higher_tf_adx(df_higher: pd.DataFrame, target_df: pd.DataFrame,
                  period: int = 14, threshold: int = 20) -> pd.Series:
    """Higher TF: ADX > 20 and +DI > -DI → bullish (uses previous completed bar)"""
    shifted = df_higher.shift(1)
    adx_series, plus_di, minus_di = compute_adx(shifted, period)
    higher_signal = pd.Series(0, index=df_higher.index)
    trending = adx_series > threshold
    higher_signal[trending & (plus_di > minus_di)] = 1
    higher_signal[trending & (minus_di > plus_di)] = -1
    return higher_signal.reindex(target_df.index, method='ffill').fillna(0)


# ═══════════════════════════════════════════════════════════════
# Group D: Volume & Order Flow (3 systems)
# ═══════════════════════════════════════════════════════════════

def obv_trend(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: OBV EMA(5) > OBV EMA(20) → bullish"""
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    obv_ema5 = obv.ewm(span=5, adjust=False).mean()
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[obv_ema5 > obv_ema20] = 1
    signal[obv_ema5 < obv_ema20] = -1
    return signal


def close_location(df: pd.DataFrame) -> pd.Series:
    """Close in upper half of bar → buying pressure, bullish"""
    bar_range = df['high'] - df['low']
    close_loc = (df['close'] - df['low']) / bar_range.replace(0, np.nan)
    signal = pd.Series(0, index=df.index)
    signal[close_loc > 0.5] = 1
    signal[close_loc < 0.5] = -1
    return signal


def cmf_bias(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow > 0 → bullish"""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
          (df['high'] - df['low']).replace(0, np.nan)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
    signal = pd.Series(0, index=df.index)
    signal[cmf > 0] = 1
    signal[cmf < 0] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Group E: Volatility & Bands (3 systems)
# ═══════════════════════════════════════════════════════════════

def bb_position(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """Price > BB middle → bullish"""
    sma = df['close'].rolling(period).mean()
    signal = pd.Series(0, index=df.index)
    signal[df['close'] > sma] = 1
    signal[df['close'] < sma] = -1
    return signal


def bb_percent_b(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.Series:
    """%B > 0.5 → bullish"""
    sma = df['close'].rolling(period).mean()
    std_dev = df['close'].rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    pct_b = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
    signal = pd.Series(0, index=df.index)
    signal[pct_b > 0.5] = 1
    signal[pct_b < 0.5] = -1
    return signal


def keltner_position(df: pd.DataFrame, period: int = 20, 
                     atr_mult: float = 2.0) -> pd.Series:
    """Price > Keltner Channel middle → bullish"""
    ema = df['close'].ewm(span=period, adjust=False).mean()
    signal = pd.Series(0, index=df.index)
    signal[df['close'] > ema] = 1
    signal[df['close'] < ema] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Group F: Statistical / Quantitative (3 systems)
# ═══════════════════════════════════════════════════════════════

def linear_slope(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Linear regression slope over last N bars > 0 → bullish"""
    x = np.arange(period)
    x_mean = x.mean()
    
    def calc_slope(window):
        if len(window) < period:
            return 0
        y = window.values
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        return num / den if den != 0 else 0
    
    slope = df['close'].rolling(period).apply(calc_slope, raw=False)
    signal = pd.Series(0, index=df.index)
    signal[slope > 0] = 1
    signal[slope < 0] = -1
    return signal


def returns_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Z-score of returns over last N bars > 0 → bullish"""
    returns = df['close'].pct_change()
    ret_mean = returns.rolling(period).mean()
    ret_std = returns.rolling(period).std()
    zscore = (returns - ret_mean) / ret_std.replace(0, np.nan)
    signal = pd.Series(0, index=df.index)
    signal[zscore > 0] = 1
    signal[zscore < 0] = -1
    return signal


def trend_consistency(df: pd.DataFrame) -> pd.Series:
    """Close > SMA(20) AND Close > Close[5] → bullish (dual confirm)"""
    sma20 = df['close'].rolling(20).mean()
    signal = pd.Series(0, index=df.index)
    bull = (df['close'] > sma20) & (df['close'] > df['close'].shift(5))
    bear = (df['close'] < sma20) & (df['close'] < df['close'].shift(5))
    signal[bull] = 1
    signal[bear] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Group G: Price Structure (3 systems)
# ═══════════════════════════════════════════════════════════════

def higher_highs(df: pd.DataFrame, lookback: int = 5, swings: int = 3) -> pd.Series:
    """Detect Higher Highs sequence → bullish (simplified swing detection)"""
    highs = df['high'].rolling(lookback, center=True).max()
    lows = df['low'].rolling(lookback, center=True).min()
    
    # A swing high: local maximum
    is_swing_high = (df['high'] == highs) & (df['high'] > df['high'].shift(1))
    is_swing_low = (df['low'] == lows) & (df['low'] < df['low'].shift(1))
    
    signal = pd.Series(0, index=df.index)
    
    # Simplified: check if last few swing points are trending up
    swing_high_mask = is_swing_high.astype(int)
    swing_low_mask = is_swing_low.astype(int)
    
    # Count recent swing highs vs swing lows bias
    recent_swing_highs = swing_high_mask.rolling(lookback * 3).sum()
    recent_swing_lows = swing_low_mask.rolling(lookback * 3).sum()
    
    signal[recent_swing_highs > recent_swing_lows] = 1
    signal[recent_swing_lows > recent_swing_highs] = -1
    
    return signal


def donchian_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Close relative to Donchian Channel midpoint → bullish if above mid"""
    high_n = df['high'].rolling(period).max()
    low_n = df['low'].rolling(period).min()
    mid = (high_n + low_n) / 2
    signal = pd.Series(0, index=df.index)
    signal[df['close'] > mid] = 1
    signal[df['close'] < mid] = -1
    return signal


def directional_movement(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """+DI > -DI → bullish"""
    _, plus_di, minus_di = compute_adx(df, period)
    signal = pd.Series(0, index=df.index)
    signal[plus_di > minus_di] = 1
    signal[plus_di < minus_di] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Group H: Alternative Perspectives (2 systems)
# ═══════════════════════════════════════════════════════════════

def ichimoku_bias(df: pd.DataFrame) -> pd.Series:
    """Price > (Tenkan + Kijun) / 2 → bullish"""
    tenkan_high = df['high'].rolling(9).max()
    tenkan_low = df['low'].rolling(9).min()
    tenkan = (tenkan_high + tenkan_low) / 2
    
    kijun_high = df['high'].rolling(26).max()
    kijun_low = df['low'].rolling(26).min()
    kijun = (kijun_high + kijun_low) / 2
    
    mid = (tenkan + kijun) / 2
    signal = pd.Series(0, index=df.index)
    signal[df['close'] > mid] = 1
    signal[df['close'] < mid] = -1
    return signal


def sma_cross_20_100(df: pd.DataFrame) -> pd.Series:
    """SMA(20) > SMA(100) → bullish (longer-term trend bias)"""
    sma20 = df['close'].rolling(20).mean()
    sma100 = df['close'].rolling(100).mean()
    signal = pd.Series(0, index=df.index)
    signal[sma20 > sma100] = 1
    signal[sma20 < sma100] = -1
    return signal


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14):
    """Compute ADX, +DI, -DI."""
    high, low, close = df['high'], df['low'], df['close']
    
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    mask_plus = (up_move > down_move) & (up_move > 0)
    mask_minus = (down_move > up_move) & (down_move > 0)
    
    plus_dm[mask_plus] = up_move[mask_plus]
    minus_dm[mask_minus] = down_move[mask_minus]
    
    atr = compute_atr(df, period)
    
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di
