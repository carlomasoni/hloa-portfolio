# hloa-portfolio





## 📂 Project Structure

<details>
<summary>Click to expand</summary>

```text
hloa-portfolio/
├── README.md
├── pyproject.toml         # dependencies & tooling config
├── src/
│   ├── hloa/              # optimiser
│   │   ├── __init__.py
│   │   ├── core.py        # algorithm implementation
│   │   └── utils.py       # RNG, population init, helpers
│   └── portfolio/         # portfolio utilities
│       ├── data.py        # returns loading & frequency handling
│       ├── estimators.py  # mean & covariance estimators
│       ├── objectives.py  # Sharpe & mean–var objectives
│       ├── constraints.py # simplex, box, turnover
│       └── frontier.py    # frontier builders & max-Sharpe wrappers
├── notebooks/
│   ├── 01_algo_sanity.ipynb    # test HLOA on toy functions
│   └── 02_frontier_demo.ipynb  # efficient frontier demo
└── tests/
    ├── test_hloa_core.py
    ├── test_constraints.py
    ├── test_objectives.py
    └── test_frontier.py
```

</details>
