"""
Data loader for event contract strategy backtesting.
Loads BTCUSDT parquet data, resamples to needed timeframes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

DATA_DIR = Path("/mnt/c/Users/12645/Sisie-Quantive/data")
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).parent / 'data'


@dataclass
class TimeframeData:
    df: pd.DataFrame
    name: str
    
    @property
    def ohlcv(self):
        return self.df[['open', 'high', 'low', 'close', 'volume']]


def load_btc_data(timeframes: list[str]) -> Dict[str, TimeframeData]:
    """
    Load BTCUSDT data for specified timeframes.
    
    Available raw: 3m, 5m, 15m, 30m, 1h, 4h
    10m is resampled from 5m.
    """
    result = {}
    
    raw_files = {
        '3m': 'binance_BTCUSDT_3m.parquet',
        '5m': 'binance_BTCUSDT_5m.parquet',
        '15m': 'binance_BTCUSDT_15m.parquet',
        '30m': 'binance_BTCUSDT_30m.parquet',
        '1h': 'binance_BTCUSDT_1h.parquet',
        '4h': 'binance_BTCUSDT_4h.parquet',
    }
    
    for tf in timeframes:
        if tf in raw_files:
            path = DATA_DIR / raw_files[tf]
            df = pd.read_parquet(path)
            result[tf] = TimeframeData(df=df, name=tf)
        elif tf == '10m':
            # Resample from 5m
            df_5m = pd.read_parquet(DATA_DIR / raw_files['5m'])
            df_10m = resample_ohlcv(df_5m, '10min')
            result[tf] = TimeframeData(df=df_10m, name=tf)
        else:
            raise ValueError(f"Unknown timeframe: {tf}")
    
    return result


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a lower frequency."""
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    return resampled.dropna()


def get_common_timerange(dfs: Dict[str, TimeframeData]) -> tuple:
    """Find the overlapping date range across all timeframes."""
    starts = [td.df.index[0] for td in dfs.values()]
    ends = [td.df.index[-1] for td in dfs.values()]
    return max(starts), min(ends)
