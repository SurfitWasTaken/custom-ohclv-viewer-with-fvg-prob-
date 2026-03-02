# FVG Probability Modeling - Data Dictionary

## Overview
This document describes the data collected during Phase -1 for the FVG Probability Modeling project.

## Data Sources

**Provider**: OANDA  
**Environment**: Practice (demo)  
**API Version**: v20 REST API  
**Collection Date**: 2026-02-14  

## Instruments

| Symbol | Name | Type | Liquidity |
|--------|------|------|-----------|
| EURUSD | Euro / US Dollar | Major | Highest |
| GBPUSD | British Pound / US Dollar | Major | High |
| USDJPY | US Dollar / Japanese Yen | Major | High |
| AUDUSD | Australian Dollar / US Dollar | Major | High |

## Timeframes

| Timeframe | OANDA Code | Candles per Day | Use Case |
|-----------|------------|-----------------|----------|
| 1H | H1 | 24 | Primary FVG detection |
| 4H | H4 | 6 | Higher timeframe context |
| 1D | D | 1 | Trend/regime identification |

## Date Ranges

- **Hourly (1H)**: 2021-02-01 to 2024-01-31 (3 years)
- **4-Hour (4H)**: 2021-02-01 to 2024-01-31 (3 years)
- **Daily (1D)**: 2019-02-01 to 2024-01-31 (5 years)

## Data Schema

### CSV Format

All data files follow this structure:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| timestamp | datetime (UTC) | Candle close time | 2021-02-01 00:00:00+00:00 |
| open | float64 | Opening price | 1.21234 |
| high | float64 | Highest price | 1.21456 |
| low | float64 | Lowest price | 1.21123 |
| close | float64 | Closing price | 1.21345 |
| volume | int64 | Tick volume | 1328 |

### Field Definitions

**timestamp**
- Format: ISO 8601 with timezone (UTC)
- Represents: Candle close time
- Index: Yes (primary key)

**open, high, low, close**
- Type: Midpoint prices (average of bid/ask)
- Precision: 5 decimal places for most pairs, 3 for JPY pairs
- Units: Quote currency per base currency

**volume**
- Type: Tick volume (number of price changes)
- Note: NOT actual traded volume (not available in retail forex)
- Use: Proxy for activity/liquidity

## File Naming Convention

```
{SYMBOL}_{TIMEFRAME}_{START_DATE}_{END_DATE}.csv
```

Examples:
- `EURUSD_1H_20210201_20240201.csv`
- `GBPUSD_4H_20210201_20240201.csv`

## Data Quality

### Validation Status

All datasets: **WARNING** (acceptable)

### Known Issues

1. **Time Gaps** (Medium severity, expected)
   - Weekend gaps: Friday 22:00 UTC to Sunday 22:00 UTC
   - Holiday gaps: Christmas (Dec 25-26), New Year (Jan 1)
   - Market closures: Occasional broker maintenance

2. **Outliers** (Low severity, legitimate)
   - High-volatility events (central bank announcements, geopolitical events)
   - Max returns: 1.74% - 4.12% (within normal forex ranges)
   - Not removed (real market data)

### Data Completeness

| Timeframe | Expected Candles | Actual Candles | Completeness |
|-----------|-----------------|----------------|--------------|
| 1H (3yr) | ~26,280 | 18,707-18,712 | ~71%* |
| 4H (3yr) | ~6,570 | 4,681 | ~71%* |
| 1D (5yr) | ~1,825 | 1,298-1,315 | ~71%* |

*Reduced by weekends (104 days/year) and holidays (~10 days/year)

## Usage Notes

### Loading Data

```python
import pandas as pd

# Load data
df = pd.read_csv('data/raw/EURUSD_1H_20210201_20240201.csv', 
                 index_col='timestamp', 
                 parse_dates=True)

# Verify
print(df.head())
print(df.info())
```

### Handling Gaps

Weekend gaps are expected and should NOT be filled:

```python
# Check for gaps
time_diff = df.index.to_series().diff()
gaps = time_diff[time_diff > pd.Timedelta(hours=2)]  # For 1H data
print(f"Found {len(gaps)} gaps")
```

### OHLC Validation

All data has been validated for:
- `high >= max(open, close)`
- `low <= min(open, close)`
- `high >= low`

No invalid candles detected.

## Metadata

### Collection Summary

See [collection_summary.json](data/raw/collection_summary.json) for:
- Exact candle counts per dataset
- Validation reports
- Issue details
- Collection timestamp

### Version Control

Data versioning not yet implemented. Current data represents:
- Collection Date: 2026-02-14
- OANDA Environment: Practice
- No subsequent updates

## Future Enhancements

1. **Database Migration**: Consider TimescaleDB for better query performance
2. **Version Tracking**: Implement data versioning with SHA256 hashes
3. **Bid/Ask Spreads**: Collect bid/ask data for spread analysis
4. **Real-time Updates**: Set up websocket feed for live data

## References

- OANDA v20 API Documentation: https://developer.oanda.com/rest-live-v20/introduction/
- Workflow Document: [fvg_probability_workflow.md](fvg_probability_workflow.md)
- Deviations Log: [deviations_log.md](deviations_log.md)
