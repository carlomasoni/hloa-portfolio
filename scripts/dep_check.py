import importlib
import sys


PACKAGES_TO_CHECK = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "sklearn",  # scikit-learn import name
    "cvxpy",
    "yaml",  # PyYAML import name
    "ecos",
    "osqp",
    "scs",
    "pytest",
]


def main() -> int:
    all_ok = True
    for package_name in PACKAGES_TO_CHECK:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "n/a")
            print(f"{package_name}: {version}")
        except Exception as exc:  # noqa: BLE001 - simple reporting script
            all_ok = False
            print(f"{package_name}: ERROR {exc}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())


