# HLOA Benchmark Testing Guide

This document provides a comprehensive guide for testing the Horned Lizard Optimization Algorithm (HLOA) implementation against standard benchmark functions.

## Overview

The benchmark testing system validates your HLOA implementation by comparing its performance against well-known optimization problems used in the original HLOA research paper. The system includes:

- **Unimodal functions**: Single global optimum (Sphere, Rosenbrock, etc.)
- **Multimodal functions**: Multiple local optima (Rastrigin, Ackley, etc.)  
- **Fixed dimension functions**: Specific dimensional problems
- **Comprehensive testing framework**: Automated benchmarking and analysis

## Quick Start

### 1. Basic Validation Test

Run the basic validation test to ensure your HLOA implementation works correctly:

```bash
python test_hloa_benchmarks.py
```

This will:
- Test basic functionality with simple functions
- Run HLOA against key benchmark functions
- Provide a summary of results

### 2. Comprehensive Benchmark Suite

For detailed performance analysis:

```bash
python -c "
import sys
sys.path.append('src')
from hloa.benchmark_runner import BenchmarkRunner
from hloa.core import HLOA_Config

config = HLOA_Config(pop_size=50, iters=200, seed=42)
runner = BenchmarkRunner(config)
results = runner.run_benchmark_suite(
    functions=['sphere', 'rastrigin', 'ackley', 'griewank'],
    dimensions=[10, 30],
    runs_per_function=5
)
runner.print_summary(results)
"
```

### 3. Interactive Notebook

For detailed analysis and visualization:

```bash
jupyter notebook notebooks/03_hloa_benchmarks.ipynb
```

## Benchmark Functions

### Unimodal Functions
- **Sphere**: `f(x) = -sum(x_i^2)` - Global optimum at origin
- **Schwefel 2.22**: `f(x) = -(sum|x_i| + prod|x_i|)` - Global optimum at origin
- **Schwefel 2.21**: `f(x) = -max|x_i|` - Global optimum at origin
- **Rosenbrock**: `f(x) = -sum(100(x_{i+1}-x_i^2)^2 + (1-x_i)^2)` - Global optimum at [1,1,...]
- **Step**: `f(x) = -sum(floor(x_i + 0.5)^2)` - Global optimum at [-0.5, -0.5, ...]
- **Quartic**: `f(x) = -sum(i*x_i^4) + noise` - Global optimum at origin

### Multimodal Functions
- **Schwefel**: `f(x) = -(418.9829*d - sum(x_i*sin(sqrt|x_i|)))` - Global optimum at [420.9687, ...]
- **Rastrigin**: `f(x) = -(10*d + sum(x_i^2 - 10*cos(2*pi*x_i)))` - Global optimum at origin
- **Ackley**: `f(x) = -(-20*exp(-0.2*sqrt(sum(x_i^2)/d)) - exp(sum(cos(2*pi*x_i))/d) + 20 + e)` - Global optimum at origin
- **Griewank**: `f(x) = -(sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1)` - Global optimum at origin
- **Penalized 1 & 2**: Complex penalty-based functions

### Fixed Dimension Functions
- **Foxholes** (2D): Complex multimodal function
- **Kowalik** (4D): Curve fitting problem
- **Six-hump Camel** (2D): Two global optima

## Performance Metrics

The benchmark system evaluates:

1. **Best Fitness**: Best solution found across all runs
2. **Mean Fitness**: Average performance across multiple runs
3. **Standard Deviation**: Consistency of results
4. **Success Rate**: Percentage of runs finding good solutions
5. **Execution Time**: Computational efficiency
6. **Convergence**: How well the algorithm approaches the global optimum

## Expected Performance

Based on the original HLOA paper, good performance should achieve:

- **Sphere**: Error < 1e-6 (very close to 0)
- **Rastrigin**: Error < 1e-2 (close to 0)
- **Ackley**: Error < 1e-6 (very close to 0)
- **Griewank**: Error < 1e-6 (very close to 0)

## Configuration Parameters

Key HLOA parameters that affect performance:

```python
config = HLOA_Config(
    pop_size=50,           # Population size (30-100 recommended)
    iters=200,             # Number of iterations (100-500 recommended)
    seed=42,               # Random seed for reproducibility
    p_mimic=0.6,           # Probability of crypsis behavior
    p_flee=0.2,            # Probability of escape behavior
    alpha_msh_threshold=0.3 # Alpha-MSH threshold for population reset
)
```

## Troubleshooting

### Poor Performance Issues

1. **Increase population size**: Try `pop_size=100`
2. **Increase iterations**: Try `iters=500`
3. **Tune parameters**: Experiment with `p_mimic` and `p_flee`
4. **Check bounds**: Ensure bounds are appropriate for the function

### Common Problems

1. **All runs failing**: Check if bounds are too restrictive
2. **Inconsistent results**: Increase number of runs or check random seed
3. **Slow convergence**: Increase population size or iterations

## Advanced Usage

### Custom Benchmark Functions

Add your own benchmark functions:

```python
from hloa.benchmarks import BenchmarkSuite

def my_function(x):
    return -np.sum(x**2, axis=1)  # Maximization version

suite = BenchmarkSuite()
suite.functions['my_function'] = {
    'func': my_function,
    'bounds': (-10, 10),
    'global_minimum': 0.0,
    'global_optimum': np.zeros(10),
    'type': 'unimodal'
}
```

### Statistical Analysis

For rigorous statistical testing:

```python
from scipy import stats

# Run multiple independent tests
results = []
for seed in range(30):  # 30 independent runs
    config = HLOA_Config(seed=seed)
    runner = BenchmarkRunner(config)
    result = runner.run_single_benchmark('sphere', runs=1)
    results.extend(result)

# Statistical analysis
fitnesses = [r.best_fitness for r in results if r.success]
print(f"Mean: {np.mean(fitnesses):.6f}")
print(f"Std: {np.std(fitnesses):.6f}")
print(f"95% CI: {stats.t.interval(0.95, len(fitnesses)-1, loc=np.mean(fitnesses), scale=stats.sem(fitnesses))}")
```

## Results Interpretation

### Success Criteria

- **Excellent**: Error < 1% of global minimum
- **Good**: Error < 5% of global minimum  
- **Acceptable**: Error < 10% of global minimum
- **Poor**: Error > 10% of global minimum

### Performance Comparison

Compare your results with:
- Original HLOA paper results
- Other metaheuristic algorithms (PSO, GA, DE, etc.)
- Theoretical bounds for the problems

## File Structure

```
src/hloa/
├── benchmarks.py          # Benchmark function implementations
├── benchmark_runner.py    # Testing framework and analysis
├── core.py               # HLOA algorithm implementation
└── ops.py                # HLOA operators

notebooks/
└── 03_hloa_benchmarks.ipynb  # Interactive analysis notebook

test_hloa_benchmarks.py   # Quick validation script
BENCHMARK_TESTING.md      # This guide
```

## References

1. Original HLOA Paper: "A novel metaheuristic inspired by horned lizard defense tactics"
2. CEC Benchmark Functions: IEEE Congress on Evolutionary Computation
3. Optimization Benchmark Literature: Various optimization algorithm papers

## Support

For questions or issues with the benchmark testing system:
1. Check the troubleshooting section above
2. Review the original HLOA paper for algorithm details
3. Examine the test results and compare with expected performance ranges
