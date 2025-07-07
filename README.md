# Worst‑Of Equity Option Pricing 

End‑to‑end toolkit that **scrapes full option chains for any equity ticker, calibrates an arbitrage‑free SABR surface, and prices a worst‑of payoff via Monte‑Carlo**.  
Originally built for a systematic‑volatility research project (2025).

---

## 🔧 Features
- Scrapes option chains across strikes and maturities for a given equity
- Fits an arbitrage-free implied volatility surface using the SABR model
- Simulates equity paths under stochastic volatility with Monte Carlo
- Prices worst-of call and computes Greek / PV surfaces under correlation shocks

## 📁 Files
- `Multi-asset.py` – Main script for scraping, calibration, simulation, and pricing

## 🧠 Model Highlights
- **SABR Calibration** with `(α, β, ρ, ν)` to implied vol surface
- Monte Carlo simulation of joint asset paths
- Worst-of payoff pricing with stress testing on correlation inputs

---

## Quick Start

```bash
# 1. Create environment
conda create -n worstof python=3.9
conda activate worstof
pip install -r requirements.txt

# 2. Example: price worst‑of (AAPL/MSFT) observed 2025‑06‑01
python Multi-asset.py \
       --tickers AAPL MSFT \
       --val_date 2025-06-01 \
       --num_paths 100000
