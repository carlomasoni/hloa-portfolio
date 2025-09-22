#!/usr/bin/env python3
"""
HLOA Implementation Validation

- HLOA is a MAXIMIZER of fitness; for minimization test functions (Sphere, Rastrigin, Ackley),
  we pass fitness = -f_true into HLOA.
- We always print/compare the TRUE objective value (f_true), i.e., best_true = -best_fitness.
- Success checks use absolute error vs the paper's global minimum (usually 0.0).
"""

# NOTE: If you've installed the repo in editable mode (`pip install -e .[dev]`),
# you do NOT need these sys.path modifications. They are left commented for portability.
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

import numpy as np
from hloa.core import HLOA, HLOA_Config
from benchmarks import BenchmarkSuite
from benchmark_runner import BenchmarkRunner


# ----------------------------
# Basic functionality (2-D)
# ----------------------------
def test_basic_functionality():
    print("Testing basic HLOA functionality...")

    # Quadratic bowl centered at (0.5, 0.5): f_true(x) = ||x - 0.5||^2
    # Our fitness must be -f_true for a maximizer.
    def quadratic_fitness(X):
        return -np.sum((X - 0.5) ** 2, axis=1)

    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=20, iters=50, seed=42)

    opt = HLOA(obj=quadratic_fitness, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    best_true = -best_fit
    print(f"  Quadratic test: best_true = {best_true:.6e}, solution = {best_sol}")
    assert best_true < 1e-1, "HLOA should find good solution for quadratic function"

    # Sphere: f_true(x) = sum(x^2); fitness = -f_true
    def sphere_fitness(X):
        return -np.sum(X ** 2, axis=1)

    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    opt = HLOA(obj=sphere_fitness, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    best_true = -best_fit
    print(f"  Sphere test: best_true = {best_true:.6e}, solution = {best_sol}")
    assert best_true < 1e-1, "HLOA should find good solution for sphere function"

    print("✓ Basic functionality tests passed!\n")


# ----------------------------
# Paper-style benchmark smoke
# ----------------------------
def test_benchmark_functions():
    print("Testing HLOA against benchmark functions...")

    suite = BenchmarkSuite()
    test_functions = ["sphere", "rastrigin", "ackley"]

    for func_name in test_functions:
        print(f"  Testing {func_name}...")

        # suite.get_function returns (f_true, bounds, info)
        func_true, bounds, func_info = suite.get_function(func_name, dim=10)

        # HLOA maximizes; wrap true objective with a minus sign
        def fitness(X, f=func_true):
            return -f(X)

        config = HLOA_Config(pop_size=30, iters=100, seed=42)
        opt = HLOA(obj=fitness, bounds=bounds, config=config)

        best_sol, best_fit, _, _ = opt.run()
        best_true = -best_fit  # convert back to true objective for reporting

        global_min = float(func_info.get("global_minimum", 0.0))
        success = abs(best_true - global_min) < 1e-3

        print(f"    Best true value: {best_true:.6e}")
        print(f"    Global minimum (paper): {global_min}")
        print(f"    Success: {success}")

    print("✓ Benchmark function tests completed!\n")


# ----------------------------
# Runner-based quick benchmark
# ----------------------------
def run_quick_benchmark():
    print("Running quick benchmark test...")

    config = HLOA_Config(
        pop_size=50,
        iters=500,
        seed=42,
        p_mimic=0.6,
        p_flee=0.2,
        alpha_msh_threshold=0.3,
    )

    runner = BenchmarkRunner(config)

    # Ensure the runner internally passes fitness = -f_true into HLOA.
    # If it does not, adjust its construction of HLOA similarly to test_benchmark_functions().
    results = runner.run_quick_test(
        functions=["sphere", "rastrigin", "ackley"],
        dimension=10,
        runs=10,
    )

    # Make sure the runner prints TRUE objective in its summary.
    # If the runner prints "fitness", it should convert to -fitness before display.
    runner.print_summary(results)

    return results


# ----------------------------
# Main
# ----------------------------
def main():
    print("HLOA Implementation Validation")
    print("=" * 40)

    try:
        test_basic_functionality()
        test_benchmark_functions()
        results = run_quick_benchmark()

        print("\n" + "=" * 40)
        print("VALIDATION COMPLETE")
        print("=" * 40)

        # Robust success aggregation without assuming a .target attribute.
        successful = 0
        total = 0
        for r in results:
            total += 1
            # Expect `best_fitness` stored as the maximized fitness value (-f_true)
            best_true = -getattr(r, "best_fitness", np.nan)

            # prefer r.global_minimum if present; otherwise fall back to 0.0
            global_min = getattr(r, "global_minimum", 0.0)

            # consider an 'error' field optional
            err_field = getattr(r, "error", None)
            ok = (abs(best_true - global_min) < 1e-3) and (err_field is None)
            successful += int(ok)

        print(f"Successful runs: {successful}/{total}")

        if successful == total:
            print(" All tests passed!")
        elif successful > total * 0.7:
            print(" Most tests passed. Your HLOA implementation is working well.")

        print("\nTo run a comprehensive benchmark suite, use:")
        print("  python -m hloa.benchmark_runner")

    except Exception as e:
        print(f" Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
