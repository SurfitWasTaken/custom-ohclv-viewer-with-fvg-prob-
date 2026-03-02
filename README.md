# FVG Probability Modeling 📈

A quantitative research project and production-ready toolset for predicting **Fair Value Gap (FVG)** fill probabilities using Survival Analysis and Machine Learning.

## 🚀 Core Functionality

This project implements a complete pipeline to:
1.  **Detect FVGs**: Identifies bullish (BISI) and bearish (SIBI) gaps across multiple timeframes.
2.  **Model Probability**: Uses a **Survival Analysis** model (`survival_model.pkl`) to estimate the *probability of mitigation* over time (e.g., "95% chance of being filled within 100 candles").
3.  **Real-time Prediction**: A V2 **Fill Predictor** that analyzes the competing "pull" between bullish and bearish gaps to establish market bias.

## 📊 Key Research Findings

*   **High Reversion Rate**: On higher timeframes (1H, 4H), over **93%** of detected FVGs are eventually touched or mitigated.
*   **The "Pull" Effect**: Unmitigated gaps act as price magnets. A cluster of high-probability bearish gaps above price typically exerts a **Bullish Pull** (ICT logic), drawing price upward for mitigation.
*   **Time Sensitivity**: Most FVGs are reached within the first **3-8 candles** of their formation.
*   **Dataset**: Validated across ~113,000 candles from 4 major pairs (EURUSD, GBPUSD, USDJPY, AUDUSD) over a 3-5 year period.

## 🖥️ Visual Dashboard

The project includes a sleek, interactive dashboard for monitoring live or historical gaps.

- **Candlestick Charts**: Real-time FVG detection and visualization.
- **Probability Bars**: Dynamic display of fill probabilities across 2, 5, 10, 20, 50, and 100 candle horizons.
- **Bias Indicators**: Automated labeling of `BULLISH PULL`, `BEARISH PULL`, or `COMPETING` biases based on active gaps.

### Start the Dashboard
```bash
source venv/bin/activate

# Offline mode (uses local CSV data)
python3 src/monitor_dashboard.py --port 5001 --offline

# Live mode (requires OANDA API credentials)
python3 src/monitor_dashboard.py --port 5001
```

## 🛠️ Project Structure

```text
fvg-probability-v2/
├── src/
│   ├── fill_predictor.py     # V2 Prediction engine (Survival Model)
│   ├── monitor_dashboard.py  # Interactive visualization
│   ├── fvg_detector.py       # Core gap identification logic
│   └── oanda_collector.py     # Resilient data acquisition
├── data/
│   ├── raw/                  # ~113k candles across 12 datasets
│   └── processed/            # FVG statistics and extracted features
├── models/                   # Serialized survival models and metadata
├── collect_data.py           # Bulk data collection script
└── scan_all_datasets.py      # Statistical FVG batch scanner
```

## 🔐 Security & Portability (GitHub Ready)

This repository has been sanitized for public use:
- **Masked Credentials**: API account IDs are masked in all console outputs.
- **Relative Paths**: All logic uses relative pathing, making the project portable across different OS/environments.
- **Ignored Secrets**: `.env` and sensitive artifacts are excluded via `.gitignore`.
- **Template**: See [.env.example](.env.example) to set up your own OANDA credentials.

## 📦 Installation

```bash
git clone https://github.com/SurfitWasTaken/custom-ohclv-viewer-with-fvg-prob-.git
cd fvg-probability-v2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---
*Disclaimer: This is a quantitative research tool, not financial advice.*
*Authored by Kallif, Claude Opus 4.6 and Gemini 3 Flash*
*Last updated: 2nd March 2026*