import numpy as np

from hloa.ops import alpha_msh, crypsis, move_to_escape, sigma, skin_lord
from portfolio.constraints import apply_bounds


def test_sigma_function():
    rng = np.random.Generator(np.random.PCG64(42))

    results = [sigma(rng) for _ in range(100)]

    assert all(result in [0, 1] for result in results)

    assert 0 in results
    assert 1 in results


def test_crypsis_basic_functionality():
    n, d = 10, 3
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 5
    max_iter = 100

    result = crypsis(X, X_best, t, max_iter)

    assert result.shape == (n, d)
    assert isinstance(result, np.ndarray)


def test_crypsis_deterministic_with_seed():
    n, d = 4, 3
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 15
    max_iter = 50

    rng1 = np.random.Generator(np.random.PCG64(456))
    result1 = crypsis(X, X_best, t, max_iter, rng=rng1)
    rng2 = np.random.Generator(np.random.PCG64(456))
    result2 = crypsis(X, X_best, t, max_iter, rng=rng2)

    np.testing.assert_array_equal(result1, result2)


def test_crypsis_edge_cases():
    n, d = 3, 2
    X = np.zeros((n, d))
    X_best = np.ones(d)
    t = 0
    max_iter = 1

    result = crypsis(X, X_best, t, max_iter)

    assert result.shape == (n, d)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_apply_bounds_simplex():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])

    result = apply_bounds(X, "simplex")

    assert np.all(result >= 0)

    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    assert result.shape == X.shape


def test_apply_bounds_box_constraints():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0, 1.0])

    result = apply_bounds(X, (lower_bounds, upper_bounds))

    assert np.all(result >= lower_bounds)
    assert np.all(result <= upper_bounds)


def test_crypsis_with_simplex_bounds():
    n, d = 5, 3
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 10
    max_iter = 100

    result = crypsis(X, X_best, t, max_iter, bounds="simplex")

    assert np.all(result >= 0)
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)


def test_skin_lord_updates_only_worst_and_respects_shape():
    n, d = 6, 4
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    idx_worst = 3
    rng = np.random.Generator(np.random.PCG64(123))

    X_new = skin_lord(X, X_best, idx_worst=idx_worst, rng=rng)

    assert X_new.shape == X.shape
    mask = np.ones(n, dtype=bool)
    mask[idx_worst] = False
    assert np.allclose(X_new[mask], X[mask])


def test_move_to_escape_shape_and_rng_determinism():
    n, d = 8, 5
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    rng1 = np.random.Generator(np.random.PCG64(777))
    rng2 = np.random.Generator(np.random.PCG64(777))

    out1 = move_to_escape(X, X_best, rng=rng1)
    out2 = move_to_escape(X, X_best, rng=rng2)
    assert out1.shape == (n, d)
    np.testing.assert_array_equal(out1, out2)


def test_alpha_msh_no_reset_returns_original():
    n, d = 6, 3
    X = np.random.random((n, d))
    fitness = np.ones(n)
    rng = np.random.Generator(np.random.PCG64(9))

    X_out, msh, changed_mask = alpha_msh(X, fitness, rng, threshold=0.0)
    assert X_out.shape == X.shape
    np.testing.assert_array_equal(X_out, X)
    assert changed_mask.shape == fitness.shape


def test_alpha_msh_resets_some_and_applies_bounds_when_requested():
    n, d = 10, 4
    X = np.random.random((n, d))
    fitness = np.linspace(0.0, 1.0, n)
    rng = np.random.Generator(np.random.PCG64(1234))
    lower = np.zeros(d)
    upper = np.ones(d)

    X_out, msh, changed_mask = alpha_msh(
        X, fitness, rng, threshold=0.8, bounds=(lower, upper)
    )

    assert X_out.shape == X.shape
    assert np.any(changed_mask)
    assert np.all(X_out >= lower)
    assert np.all(X_out <= upper)
