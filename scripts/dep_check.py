import importlib, sys
pkgs = ["numpy","scipy","pandas","matplotlib","sklearn","cvxpy","yaml","ecos","osqp","scs","pytest"]
ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "n/a")
        print(f"{p}: {v}")
    except Exception as e:
        ok = False
        print(f"{p}: ERROR {e}")
sys.exit(0 if ok else 1)
