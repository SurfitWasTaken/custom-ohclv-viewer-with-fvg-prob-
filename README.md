# FVG Probability Modeling - Phase -1 Complete ✅

## Quick Start

### Activate Environment
```bash
source venv/bin/activate
```

### View Collected Data
```bash
ls -lh data/raw/
```

### Load Data in Python
```python
import pandas as pd

# Load EURUSD 1H data
df = pd.read_csv('data/raw/EURUSD_1H_20210201_20240201.csv', 
                 index_col='timestamp', parse_dates=True)
print(df.head())
```

## Project Structure

```
fvg-probability/
├── data/
│   ├── raw/                    # 12 CSV files (4 pairs × 3 timeframes)
│   └── processed/              # (empty - for Phase 0+)
├── src/
│   ├── oanda_collector.py      # OANDA API data collector
│   └── data_validator.py       # Data quality validation
├── venv/                       # Python virtual environment
├── collect_data.py             # Main data collection script
├── requirements.txt            # Python dependencies
├── DATA_DICTIONARY.md          # Data schema and documentation
└── fvg_probability_workflow.md # Original workflow document
```

## Data Summary

**Collected**: 12 datasets (4 currency pairs × 3 timeframes)  
**Total Candles**: ~113,000  
**Date Range**: Feb 2021 - Jan 2024 (3 years intraday, 5 years daily)  
**Quality**: All datasets validated ✅

| Pair | 1H | 4H | Daily |
|------|----|----|-------|
| EURUSD | 18,712 | 4,681 | 1,299 |
| GBPUSD | 18,711 | 4,681 | 1,303 |
| USDJPY | 18,712 | 4,681 | 1,315 |
| AUDUSD | 18,707 | 4,681 | 1,298 |

## Next Steps

✅ **Phase -1 Complete**: Data Acquisition  
✅ **Phase 0 Complete**: FVG Identification & Validation

Ready for **Phase 1: Feature Engineering**

### Phase 0 Results

**FVG Detection Summary**:
- 19,840 FVGs detected across all datasets
- 20.08% FVG rate (1 in 5 candles)
- 100% reversion rate (all touched within 1 candle)
- 50 sample visualizations generated

**Key Finding**: High FVG frequency suggests current definition captures micro-gaps. Phase 1 will add filtering for meaningful gaps.

See workflow document for Phase 1 details: [fvg_probability_workflow.md](fvg_probability_workflow.md#phase-1-feature-engineering-week-2)

## Documentation

- **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** - Data schema and field definitions
- **[deviations_log.md](../.gemini/antigravity/brain/c8ebe8c1-12e4-4f0e-a43f-8846abf9720f/deviations_log.md)** - Workflow deviations and impact analysis
- **[walkthrough.md](../.gemini/antigravity/brain/c8ebe8c1-12e4-4f0e-a43f-8846abf9720f/walkthrough.md)** - Complete Phase -1 walkthrough
- **[collection_summary.json](data/raw/collection_summary.json)** - Detailed collection results

## Key Deviations from Workflow

✅ **Using OANDA API instead of MT5** - Better API, production-ready  
✅ **CSV storage instead of TimescaleDB** - Simpler for development  
✅ **Date-based chunking** - Handles OANDA's 5000 candle limit  

**Impact on Future Phases**: None - all deviations are improvements or neutral

## Dependencies

```
Python 3.13
oandapyV20==0.7.2
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
```

Install: `pip install -r requirements.txt`
