from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple

import numpy as np


class BenchmarkFunction(ABC):
    def __init__(self, name: str, dim: int, bounds: Tuple[float, float]):
        self.name = name
        self.dim = dim
        self.bounds = bounds
        self.global_minimum = None
        self.global_optimum = None

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.full(self.dim, self.bounds[0])
        ub = np.full(self.dim, self.bounds[1])
        return lb, ub


class UnimodalFunctions:
    @staticmethod
    def sphere(x: np.ndarray) -> np.ndarray:
        return -np.sum(x**2, axis=1)

    @staticmethod
    def schwefel_2_22(x: np.ndarray) -> np.ndarray:
        return -(np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1))

    @staticmethod
    def schwefel_2_21(x: np.ndarray) -> np.ndarray:
        return -np.max(np.abs(x), axis=1)

    @staticmethod
    def rosenbrock(x: np.ndarray) -> np.ndarray:
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        return -np.sum(100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2, axis=1)

    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        return -np.sum((np.floor(x + 0.5)) ** 2, axis=1)

    @staticmethod
    def quartic(x: np.ndarray) -> np.ndarray:
        i = np.arange(1, x.shape[1] + 1)
        return -(np.sum(i * x**4, axis=1) + np.random.random(x.shape[0]))


class MultimodalFunctions:
    @staticmethod
    def schwefel(x: np.ndarray) -> np.ndarray:
        return -(418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1))

    @staticmethod
    def rastrigin(x: np.ndarray) -> np.ndarray:
        return -(10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1))

    @staticmethod
    def ackley(x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(np.cos(2 * np.pi * x), axis=1)
        return -(-20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e)

    @staticmethod
    def griewank(x: np.ndarray) -> np.ndarray:
        sum_sq = np.sum(x**2, axis=1)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
        return -(sum_sq / 4000 - prod_cos + 1)

    @staticmethod
    def penalized_1(x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        y = 1 + 0.25 * (x + 1)

        term1 = np.sum(
            (y[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[:, 1:]) ** 2), axis=1
        )
        term2 = 10 * np.sin(np.pi * y[:, 0]) ** 2
        term3 = (y[:, -1] - 1) ** 2

        u = np.zeros_like(x)
        u[x > 10] = 100 * (x[x > 10] - 10) ** 4
        u[x < -10] = 100 * (-x[x < -10] - 10) ** 4

        return -((np.pi / n) * (term1 + term2 + term3) + np.sum(u, axis=1))

    @staticmethod
    def penalized_2(x: np.ndarray) -> np.ndarray:
        term1 = 0.1 * (
            np.sin(3 * np.pi * x[:, 0]) ** 2
            + np.sum(
                (x[:, :-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[:, 1:]) ** 2), axis=1
            )
            + (x[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[:, -1]) ** 2)
        )

        u = np.zeros_like(x)
        u[x > 5] = 100 * (x[x > 5] - 5) ** 4
        u[x < -5] = 100 * (-x[x < -5] - 5) ** 4

        return -(term1 + np.sum(u, axis=1))


class FixedDimensionFunctions:
    @staticmethod
    def foxholes(x: np.ndarray) -> np.ndarray:
        a = np.array(
            [
                [
                    -32,
                    -16,
                    0,
                    16,
                    32,
                    -32,
                    -16,
                    0,
                    16,
                    32,
                    -32,
                    -16,
                    0,
                    16,
                    32,
                    -32,
                    -16,
                    0,
                    16,
                    32,
                    -32,
                    -16,
                    0,
                    16,
                    32,
                ],
                [
                    -32,
                    -32,
                    -32,
                    -32,
                    -32,
                    -16,
                    -16,
                    -16,
                    -16,
                    -16,
                    0,
                    0,
                    0,
                    0,
                    0,
                    16,
                    16,
                    16,
                    16,
                    16,
                    32,
                    32,
                    32,
                    32,
                    32,
                ],
            ]
        )

        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            sum_val = 0
            for j in range(25):
                sum_val += 1 / (
                    j + 1 + (x[i, 0] - a[0, j]) ** 6 + (x[i, 1] - a[1, j]) ** 6
                )
            result[i] = 1 / (1 / 500 + sum_val)
        return result

    @staticmethod
    def kowalik(x: np.ndarray) -> np.ndarray:
        a = np.array(
            [
                0.1957,
                0.1947,
                0.1735,
                0.1600,
                0.0844,
                0.0627,
                0.0456,
                0.0342,
                0.0323,
                0.0235,
                0.0246,
            ]
        )
        b = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])

        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            sum_val = 0
            for j in range(11):
                sum_val += (
                    a[j]
                    - (x[i, 0] * (b[j] ** 2 + b[j] * x[i, 1]))
                    / (b[j] ** 2 + b[j] * x[i, 2] + x[i, 3])
                ) ** 2
            result[i] = -sum_val
        return result

    @staticmethod
    def six_hump_camel(x: np.ndarray) -> np.ndarray:
        return -(
            (4 - 2.1 * x[:, 0] ** 2 + x[:, 0] ** 4 / 3) * x[:, 0] ** 2
            + x[:, 0] * x[:, 1]
            + (-4 + 4 * x[:, 1] ** 2) * x[:, 1] ** 2
        )


class BenchmarkSuite:
    def __init__(self):
        self.functions = {}
        self._setup_functions()

    def _setup_functions(self):
        self.functions["sphere"] = {
            "func": UnimodalFunctions.sphere,
            "bounds": (-100, 100),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "unimodal",
        }

        self.functions["schwefel_2_22"] = {
            "func": UnimodalFunctions.schwefel_2_22,
            "bounds": (-10, 10),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "unimodal",
        }

        self.functions["schwefel_2_21"] = {
            "func": UnimodalFunctions.schwefel_2_21,
            "bounds": (-100, 100),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "unimodal",
        }

        self.functions["rosenbrock"] = {
            "func": UnimodalFunctions.rosenbrock,
            "bounds": (-30, 30),
            "global_minimum": 0.0,
            "global_optimum": np.ones(30),
            "type": "unimodal",
        }

        self.functions["step"] = {
            "func": UnimodalFunctions.step,
            "bounds": (-100, 100),
            "global_minimum": 0.0,
            "global_optimum": np.full(30, -0.5),
            "type": "unimodal",
        }

        self.functions["quartic"] = {
            "func": UnimodalFunctions.quartic,
            "bounds": (-1.28, 1.28),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "unimodal",
        }

        self.functions["schwefel"] = {
            "func": MultimodalFunctions.schwefel,
            "bounds": (-500, 500),
            "global_minimum": 0.0,
            "global_optimum": np.full(30, 420.9687),
            "type": "multimodal",
        }

        self.functions["rastrigin"] = {
            "func": MultimodalFunctions.rastrigin,
            "bounds": (-5.12, 5.12),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "multimodal",
        }

        self.functions["ackley"] = {
            "func": MultimodalFunctions.ackley,
            "bounds": (-32, 32),
            "global_minimum": 0.0,
            "type": "multimodal",
        }

        self.functions["griewank"] = {
            "func": MultimodalFunctions.griewank,
            "bounds": (-600, 600),
            "global_minimum": 0.0,
            "global_optimum": np.zeros(30),
            "type": "multimodal",
        }

        self.functions["penalized_1"] = {
            "func": MultimodalFunctions.penalized_1,
            "bounds": (-50, 50),
            "global_minimum": 0.0,
            "global_optimum": np.ones(30),
            "type": "multimodal",
        }

        self.functions["penalized_2"] = {
            "func": MultimodalFunctions.penalized_2,
            "bounds": (-50, 50),
            "global_minimum": 0.0,
            "global_optimum": np.ones(30),
            "type": "multimodal",
        }

        self.functions["foxholes"] = {
            "func": FixedDimensionFunctions.foxholes,
            "bounds": (-65.536, 65.536),
            "global_minimum": 0.998,
            "global_optimum": np.array([-32, -32]),
            "type": "fixed_dimension",
            "dim": 2,
        }

        self.functions["kowalik"] = {
            "func": FixedDimensionFunctions.kowalik,
            "bounds": (-5, 5),
            "global_minimum": 0.0,
            "global_optimum": np.array([0.1928, 0.1908, 0.1231, 0.1358]),
            "type": "fixed_dimension",
            "dim": 4,
        }

        self.functions["six_hump_camel"] = {
            "func": FixedDimensionFunctions.six_hump_camel,
            "bounds": (-5, 5),
            "global_minimum": 1.0316,
            "global_optimum": np.array([0.0898, -0.7126]),
            "type": "fixed_dimension",
            "dim": 2,
        }

    def get_function(
        self, name: str, dim: int = 30
    ) -> Tuple[Callable, Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found in benchmark suite")

        func_info = self.functions[name]
        func = func_info["func"]

        if func_info["type"] == "fixed_dimension":
            actual_dim = func_info["dim"]
        else:
            actual_dim = dim

        lb = np.full(actual_dim, func_info["bounds"][0])
        ub = np.full(actual_dim, func_info["bounds"][1])

        def wrapper(x: np.ndarray) -> np.ndarray:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return func(x)

        return wrapper, (lb, ub), func_info

    def list_functions(self) -> list:
        return list(self.functions.keys())

    def get_functions_by_type(self, func_type: str) -> list:
        return [
            name for name, info in self.functions.items() if info["type"] == func_type
        ]


def create_benchmark_suite() -> BenchmarkSuite:
    return BenchmarkSuite()
