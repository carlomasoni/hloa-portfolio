#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from hloa.core import HLOA, HLOA_Config
from hloa.benchmarks import BenchmarkSuite
from hloa.benchmark_runner import BenchmarkRunner


def test_basic_functionality():
    print("Testing basic HLOA functionality...")
    
    def quadratic(x):
        return -np.sum((x - 0.5)**2, axis=1)
    
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=20, iters=50, seed=42)
    
    opt = HLOA(obj=quadratic, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    
    print(f"  Quadratic test: best_fitness = {best_fit:.6f}, solution = {best_sol}")
    assert best_fit > -0.1, "HLOA should find good solution for quadratic function"
    
    def sphere(x):
        return -np.sum(x**2, axis=1)
    
    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    opt = HLOA(obj=sphere, bounds=bounds, config=config)
    best_sol, best_fit, _, _ = opt.run()
    
    print(f"  Sphere test: best_fitness = {best_fit:.6f}, solution = {best_sol}")
    assert best_fit > -0.1, "HLOA should find good solution for sphere function"
    
    print("✓ Basic functionality tests passed!\n")


def test_benchmark_functions():
    print("Testing HLOA against benchmark functions...")
    
    suite = BenchmarkSuite()
    test_functions = ['sphere', 'rastrigin', 'ackley']
    
    for func_name in test_functions:
        print(f"  Testing {func_name}...")
        
        func, bounds, func_info = suite.get_function(func_name, dim=10)
        
        config = HLOA_Config(pop_size=30, iters=100, seed=42)
        opt = HLOA(obj=func, bounds=bounds, config=config)
        
        best_sol, best_fit, _, _ = opt.run()
        
        print(f"    Best fitness: {best_fit:.6f}")
        print(f"    Global minimum: {func_info['global_minimum']}")
        print(f"    Success: {abs(best_fit - func_info['global_minimum']) < 1e-3}")
    
    print("✓ Benchmark function tests completed!\n")


def run_quick_benchmark():
    print("Running quick benchmark test...")
    
    config = HLOA_Config(
        pop_size=30,
        iters=100,
        seed=42,
        p_mimic=0.6,
        p_flee=0.2,
        alpha_msh_threshold=0.3
    )
    
    runner = BenchmarkRunner(config)
    
    results = runner.run_quick_test(
        functions=['sphere', 'rastrigin', 'ackley'],
        dimension=10,
        runs=3
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
        
        successful = sum(1 for r in results if r.success and r.error is None)
        total = len(results)
        
        print(f"Successful runs: {successful}/{total}")
        
        if successful == total:
            print("🎉 All tests passed! Your HLOA implementation appears to be working correctly.")
        elif successful > total * 0.7:
            print("✅ Most tests passed. Your HLOA implementation is working well.")
        else:
            print("⚠️  Some tests failed. You may want to review your implementation.")
        
        print("\nTo run a comprehensive benchmark suite, use:")
        print("  python -m hloa.benchmark_runner")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())