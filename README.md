# HLOA Portfolio Optimization

Portfolio optimization using the Horned Lizard Optimization Algorithm (HLOA) and QUBO (Quadratic Unconstrained Binary Optimization) methods for EuroStoxx 50 stocks.

## Features

- **HLOA Optimization**: Continuous weight optimization using evolutionary algorithms
- **QUBO Optimization**: Binary asset selection with equal weighting
- **EuroStoxx 50 Support**: Optimized for European markets
- **Risk-Free Rate Calculation**: Automatic market data integration
- **Sharpe Ratio Maximization**: Risk-adjusted return optimization

## Project Structure

```
hloa-portfolio/
├── src/                    # Source code
│   ├── hloa/              # HLOA algorithm implementation
│   └── portfolio/         # Portfolio optimization modules
├── scripts/               # Executable scripts
├── examples/              # Example notebooks and benchmarks
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install system-wide on Arch Linux
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Quick Start

```bash
# Run HLOA optimization for EuroStoxx 50
python scripts/run_hloa_optimization.py

# Compare HLOA vs QUBO methods
python scripts/benchmark_optimization.py

# Run comprehensive tests
python scripts/test_optimization.py
```

## Usage

### HLOA Optimization
```python
from portfolio.frontier import optimize_portfolio_sharpe

results = optimize_portfolio_sharpe(
    time_period_days=2000,
    include_eurostoxx=True,
    risk_free_rate=None,
    currency='EUR'
)
```

### QUBO Optimization
```python
from portfolio.qubo_optimization import optimize_portfolio_qubo

results = optimize_portfolio_qubo(
    time_period_days=2000,
    include_eurostoxx=True,
    max_assets=10,
    method='simulated_annealing'
)
```

## Results

The optimization provides:
- Optimal portfolio weights
- Sharpe ratio
- Expected return and volatility
- Risk-free rate
- Diversification metrics

## Dependencies

- numpy >= 1.26.0
- pandas >= 0.19
- matplotlib >= 3.0
- yfinance >= 0.2.0
- PyPortfolioOpt >= 1.5.0
- scipy >= 1.3

## License

MIT License



hloa-portfolio/
├── src/                    # Source code
│   ├── hloa/              # HLOA algorithm implementation
│   └── portfolio/         # Portfolio optimization modules
├── scripts/               # Executable scripts
│   ├── run_hloa_optimization.py
│   ├── benchmark_optimization.py
│   ├── test_optimization.py
│   ├── test_hloa_benchmarks.py
│   ├── dep_check.py
│   └── README.md
├── examples/              # Example notebooks and benchmarks
│   ├── benchmark_runner.py
│   ├── benchmarks.py
│   ├── hloa_benchmarks.ipynb
│   └── README.md
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
│   └── BENCHMARK_TESTING.md
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Updated project documentation