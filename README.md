# Financial Systems Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CCejudo98/financial-systems-lab/blob/main/notebooks/hrp_demo.ipynb)

A modular quantitative engine for asset allocation and risk management using **Hierarchical Risk Parity (HRP)**, **Random Matrix Theory (RMT)**, and **Graph Theory**.

## Key Features

- **Noise Filtering (RMT):** Denoises empirical correlation matrices using Marchenko-Pastur bounds.
- **Hierarchical Clustering:** Applies angular distance metrics $d_{i,j} = \sqrt{2(1 - \rho_{i,j})}$ and Ward linkage to handle multicollinearity without matrix inversion.
- **Minimum Spanning Tree (MST):** Constructs graph representations to visualize asset correlation networks and systemic risk topology.

## Architecture

```text
financial-systems-lab/
├── hrp_engine.py       # Core allocation and graph engine
├── requirements.txt    # Environment dependencies
└── README.md           # Documentation

## Quick Start

```python
import yfinance as yf
from hrp_engine import HierarchicalRiskParity

# Download asset prices
tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM', 'NVDA']
data = yf.download(tickers, start='2022-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Run HRP Engine
engine = HierarchicalRiskParity(returns)
weights = engine.allocate()
print(weights)
