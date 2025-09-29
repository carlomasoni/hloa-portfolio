#!/usr/bin/env python3

import numpy as np
from hloa.core import HLOA, HLOA_Config
from benchmarks import BenchmarkSuite
from benchmark_runner import BenchmarkRunner


def test_basic_functionality():
    print("Testing basic HLOA functionality...")

    def quadratic_fitness(X):
        return -np.sum((X - 0.5) ** 2, axis=1)

    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=20, iters=50, seed=42)

    opt = HLOA(obj=quadratic_fitness, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    best_true = -best_fit
    print(f"  Quadratic test: best_true = {best_true:.6e}, solution = {best_sol}")
    assert best_true < 1e-1, "HLOA should find good solution for quadratic function"

    def sphere_fitness(X):
        return -np.sum(X ** 2, axis=1)

    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    opt = HLOA(obj=sphere_fitness, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    best_true = -best_fit
    print(f"  Sphere test: best_true = {best_true:.6e}, solution = {best_sol}")
    assert best_true < 1e-1, "HLOA should find good solution for sphere function"

    print("✓ Basic functionality tests passed!\n")


def test_benchmark_functions():
    print("Testing HLOA against benchmark functions...")

    suite = BenchmarkSuite()
    test_functions = ["sphere", "rastrigin", "ackley"]

    for func_name in test_functions:
        print(f"  Testing {func_name}...")

        func_true, bounds, func_info = suite.get_function(func_name, dim=10)
        def fitness(X, f=func_true):
            return -f(X)

        config = HLOA_Config(pop_size=30, iters=100, seed=42)
        opt = HLOA(obj=fitness, bounds=bounds, config=config)

        best_sol, best_fit, _, _ = opt.run()
        best_true = -best_fit

        global_min = float(func_info.get("global_minimum", 0.0))
        success = abs(best_true - global_min) < 1e-3

        print(f"    Best true value: {best_true:.6e}")
        print(f"    Global minimum (paper): {global_min}")
        print(f"    Success: {success}")

    print("✓ Benchmark function tests completed!\n")


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

    results = runner.run_quick_test(
        functions=["sphere", "rastrigin", "ackley"],
        dimension=10,
        runs=10,
    )

    runner.print_summary(results)

    return results


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

        successful = 0
        total = 0
        for r in results:
            total += 1
            best_true = -getattr(r, "best_fitness", np.nan)

            global_min = getattr(r, "global_minimum", 0.0)

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
