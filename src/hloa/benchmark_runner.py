import json
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from hloa.benchmarks import BenchmarkSuite
from hloa.core import HLOA, HLOA_Config


@dataclass
class BenchmarkResult:
    function_name: str
    dimension: int
    best_fitness: float
    best_solution: np.ndarray
    convergence_history: List[float]
    execution_time: float
    iterations: int
    population_size: int
    seed: int
    success: bool
    error: Optional[str] = None


class BenchmarkRunner:
    def __init__(self, config: Optional[HLOA_Config] = None):
        self.config = config or HLOA_Config()
        self.benchmark_suite = BenchmarkSuite()
        self.results: List[BenchmarkResult] = []

    def run_single_benchmark(
        self,
        function_name: str,
        dimension: int = 30,
        runs: int = 1,
        verbose: bool = True,
    ) -> List[BenchmarkResult]:
        if verbose:
            print(f"Running benchmark: {function_name} (dim={dimension})")

        try:
            func, bounds, func_info = self.benchmark_suite.get_function(
                function_name, dimension
            )
        except ValueError as e:
            print(f"Error: {e}")
            return []

        results = []

        for run in range(runs):
            if verbose and runs > 1:
                print(f"  Run {run + 1}/{runs}")

            config = HLOA_Config(
                pop_size=self.config.pop_size,
                iters=self.config.iters,
                seed=self.config.seed + run if self.config.seed else None,
                p_mimic=self.config.p_mimic,
                p_flee=self.config.p_flee,
                alpha_msh_threshold=self.config.alpha_msh_threshold,
            )

            opt = HLOA(obj=func, bounds=bounds, config=config)

            start_time = time.time()
            try:
                best_solution, best_fitness, final_pop, final_fitness = opt.run()
                execution_time = time.time() - start_time

                global_min = func_info["global_minimum"]
                success = (
                    abs(best_fitness - global_min) <= abs(global_min * 0.01)
                    if global_min != 0
                    else best_fitness < 1e-6
                )

                result = BenchmarkResult(
                    function_name=function_name,
                    dimension=dimension,
                    best_fitness=best_fitness,
                    best_solution=best_solution,
                    convergence_history=[],
                    execution_time=execution_time,
                    iterations=config.iters,
                    population_size=config.pop_size,
                    seed=config.seed,
                    success=success,
                )

                results.append(result)

            except Exception as e:
                result = BenchmarkResult(
                    function_name=function_name,
                    dimension=dimension,
                    best_fitness=float("inf"),
                    best_solution=np.array([]),
                    convergence_history=[],
                    execution_time=0.0,
                    iterations=config.iters,
                    population_size=config.pop_size,
                    seed=config.seed,
                    success=False,
                    error=str(e),
                )
                results.append(result)
                if verbose:
                    print(f"    Error: {e}")

        return results

    def run_benchmark_suite(
        self,
        functions: Optional[List[str]] = None,
        dimensions: List[int] = [10, 30],
        runs_per_function: int = 5,
        verbose: bool = True,
    ) -> List[BenchmarkResult]:
        if functions is None:
            functions = [
                "sphere",
                "schwefel_2_22",
                "schwefel_2_21",
                "rosenbrock",
                "step",
                "quartic",
                "schwefel",
                "rastrigin",
                "ackley",
                "griewank",
                "penalized_1",
                "penalized_2",
            ]

        all_results = []

        if verbose:
            print(f"Running benchmark suite with {len(functions)} functions")
            print(f"Dimensions: {dimensions}")
            print(f"Runs per function: {runs_per_function}")
            print("-" * 50)

        for func_name in functions:
            for dim in dimensions:
                func_info = self.benchmark_suite.functions[func_name]
                if func_info["type"] == "fixed_dimension" and dim != func_info["dim"]:
                    continue

                results = self.run_single_benchmark(
                    func_name, dim, runs_per_function, verbose
                )
                all_results.extend(results)

        self.results.extend(all_results)
        return all_results

    def run_quick_test(
        self,
        functions: List[str] = ["sphere", "rastrigin", "ackley"],
        dimension: int = 10,
        runs: int = 3,
    ) -> List[BenchmarkResult]:
        print("Running quick benchmark test...")
        print(f"Functions: {functions}")
        print(f"Dimension: {dimension}")
        print(f"Runs: {runs}")
        print("-" * 30)

        results = []
        for func_name in functions:
            func_results = self.run_single_benchmark(
                func_name, dimension, runs, verbose=True
            )
            results.extend(func_results)

        return results

    def analyze_results(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> pd.DataFrame:
        if results is None:
            results = self.results

        if not results:
            return pd.DataFrame()

        data = []
        for result in results:
            data.append(
                {
                    "function": result.function_name,
                    "dimension": result.dimension,
                    "best_fitness": result.best_fitness,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "seed": result.seed,
                    "error": result.error,
                }
            )

        df = pd.DataFrame(data)

        summary = (
            df.groupby(["function", "dimension"])
            .agg(
                {
                    "best_fitness": ["mean", "std", "min", "max"],
                    "execution_time": ["mean", "std"],
                    "success": "mean",
                }
            )
            .round(6)
        )

        return summary

    def print_summary(self, results: Optional[List[BenchmarkResult]] = None):
        if results is None:
            results = self.results

        if not results:
            print("No results to summarize.")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)

        by_function = {}
        for result in results:
            key = f"{result.function_name}_d{result.dimension}"
            if key not in by_function:
                by_function[key] = []
            by_function[key].append(result)

        for key, func_results in by_function.items():
            print(f"\n{key.upper()}:")
            print("-" * 40)

            successful_results = [
                r for r in func_results if r.success and r.error is None
            ]

            if successful_results:
                fitnesses = [r.best_fitness for r in successful_results]
                times = [r.execution_time for r in successful_results]

                print(
                    f"  Successful runs: {len(successful_results)}/{len(func_results)}"
                )
                print(f"  Best fitness: {min(fitnesses):.6f}")
                mean_fit = np.mean(fitnesses)
                std_fit = np.std(fitnesses)
                print(f"  Mean fitness: {mean_fit:.6f} ± {std_fit:.6f}")
                print(f"  Mean time: {np.mean(times):.3f}s")

                func_name = func_results[0].function_name
                func_info = self.benchmark_suite.functions[func_name]
                global_min = func_info["global_minimum"]
                best_achieved = min(fitnesses)

                if global_min != 0:
                    error_percent = (
                        abs(best_achieved - global_min) / abs(global_min) * 100
                    )
                    print(f"  Error from global min: {error_percent:.2f}%")
                else:
                    print(f"  Error from global min: {best_achieved:.2e}")
            else:
                print(f"  No successful runs out of {len(func_results)}")
                if func_results:
                    errors = [r.error for r in func_results if r.error]
                    if errors:
                        print(f"  Common errors: {set(errors)}")

    def save_results(
        self, filename: str, results: Optional[List[BenchmarkResult]] = None
    ):
        if results is None:
            results = self.results

        data = []
        for result in results:
            data.append(
                {
                    "function_name": result.function_name,
                    "dimension": result.dimension,
                    "best_fitness": float(result.best_fitness),
                    "best_solution": result.best_solution.tolist(),
                    "execution_time": result.execution_time,
                    "iterations": result.iterations,
                    "population_size": result.population_size,
                    "seed": result.seed,
                    "success": result.success,
                    "error": result.error,
                }
            )

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filename}")

    def compare_with_paper_results(
        self, results: Optional[List[BenchmarkResult]] = None
    ):
        if results is None:
            results = self.results

        print("\n" + "=" * 60)
        print("COMPARISON WITH HLOA PAPER RESULTS")
        print("=" * 60)
        print("Note: This is a basic comparison. For detailed analysis,")
        print("refer to the original HLOA paper results.")
        print("-" * 60)

        expected_ranges = {
            "sphere": (1e-10, 1e-6),
            "rastrigin": (1e-2, 1e1),
            "ackley": (1e-10, 1e-6),
            "griewank": (1e-10, 1e-6),
            "rosenbrock": (1e-2, 1e2),
            "schwefel": (1e-2, 1e2),
        }

        for result in results:
            if result.function_name in expected_ranges and result.success:
                expected_min, expected_max = expected_ranges[result.function_name]
                if expected_min <= result.best_fitness <= expected_max:
                    status = "✓ GOOD"
                elif result.best_fitness < expected_min:
                    status = "✓ EXCELLENT"
                else:
                    status = "⚠ NEEDS IMPROVEMENT"

                func_name = result.function_name
                dim = result.dimension
                fitness = result.best_fitness
                print(f"{func_name} (d={dim}): {fitness:.2e} {status}")


def run_comprehensive_benchmark():
    print("Horned Lizard Optimization Algorithm - Benchmark Test Suite")
    print("=" * 60)

    config = HLOA_Config(
        pop_size=50,
        iters=100,
        seed=42,
        p_mimic=0.6,
        p_flee=0.2,
        alpha_msh_threshold=0.3,
    )

    runner = BenchmarkRunner(config)

    print("\n1. Running Quick Test...")
    quick_results = runner.run_quick_test(
        functions=["sphere", "rastrigin", "ackley"], dimension=10, runs=3
    )

    runner.print_summary(quick_results)

    print("\n" + "=" * 60)
    response = input("Run full benchmark suite? (y/n): ").lower().strip()

    if response == "y":
        print("\n2. Running Full Benchmark Suite...")
        full_results = runner.run_benchmark_suite(
            functions=["sphere", "rastrigin", "ackley", "griewank", "rosenbrock"],
            dimensions=[10, 30],
            runs_per_function=5,
        )

        runner.print_summary(full_results)
        runner.compare_with_paper_results(full_results)

        runner.save_results("hloa_benchmark_results.json", full_results)

    return runner


if __name__ == "__main__":
    runner = run_comprehensive_benchmark()
