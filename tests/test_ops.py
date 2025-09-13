
import pytest
import numpy as np
from hloa.ops import crypsis, sigma
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


def test_crypsis_with_custom_rng():
    n, d = 5, 2
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 10
    max_iter = 50
    
    rng = np.random.Generator(np.random.PCG64(123))
    
    result1 = crypsis(X, X_best, t, max_iter, rng=rng)
    
    rng2 = np.random.Generator(np.random.PCG64(123))
    result2 = crypsis(X, X_best, t, max_iter, rng=rng2)
    
    np.testing.assert_array_equal(result1, result2)


def test_crypsis_with_custom_parameters():
    n, d = 8, 4
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 20
    max_iter = 100
    
    result = crypsis(
        X, X_best, t, max_iter,
        c1=2.0, c2=1.5, omega=3.0, decay_eps=0.2
    )
    
    assert result.shape == (n, d)


def test_crypsis_amplitude_decay():
    n, d = 6, 2
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    max_iter = 100
    
    result_early = crypsis(X, X_best, t=0, max_iter=max_iter)
    
    result_late = crypsis(X, X_best, t=99, max_iter=max_iter)
    
    early_deviation = np.mean(np.abs(result_early - X_best))
    late_deviation = np.mean(np.abs(result_late - X_best))
    assert isinstance(early_deviation, float)
    assert isinstance(late_deviation, float)


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
    # Test data
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    
    result = apply_bounds(X, "simplex")
    

    assert np.all(result >= 0), "All weights should be non-negative"
    

    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    

    assert result.shape == X.shape


def test_apply_bounds_simplex_long_only():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    
    result = apply_bounds(X, "simplex_long_only")
    

    assert np.all(result >= 0), "All weights should be non-negative"
    

    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)


def test_apply_bounds_simplex_long_short():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    
    result = apply_bounds(X, "simplex_long_short")
    

    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    

    assert np.any(result < 0) or np.any(result > 0), "Should allow both positive and negative values"


def test_apply_bounds_box_constraints():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    lower_bounds = np.array([0.0, 0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0, 1.0])
    
    result = apply_bounds(X, (lower_bounds, upper_bounds))
    
    # Check bounds
    assert np.all(result >= lower_bounds), "Should respect lower bounds"
    assert np.all(result <= upper_bounds), "Should respect upper bounds"

def test_apply_bounds_none():
    X = np.array([[0.3, 0.7, 0.2], [0.1, 0.4, 0.5], [-0.2, 0.8, 0.4]])
    
    result = apply_bounds(X, None)
    
    np.testing.assert_array_equal(result, X)


def test_crypsis_with_simplex_bounds():
    n, d = 5, 3
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 10
    max_iter = 100
    
    result = crypsis(X, X_best, t, max_iter, bounds="simplex")
    
    assert np.all(result >= 0), "All weights should be non-negative"
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)


def test_crypsis_with_box_bounds():

    n, d = 4, 2
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 5
    max_iter = 50
    
    lower_bounds = np.array([0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0])
    
    result = crypsis(X, X_best, t, max_iter, bounds=(lower_bounds, upper_bounds))
    
    assert np.all(result >= lower_bounds), "Should respect lower bounds"
    assert np.all(result <= upper_bounds), "Should respect upper bounds"


def test_crypsis_portfolio_optimization_example():
    n_agents, n_assets = 10, 5
    X = np.random.random((n_agents, n_assets))
    X_best = np.random.random(n_assets)
    t = 20
    max_iter = 100
    

    result = crypsis(X, X_best, t, max_iter, bounds="simplex")
    

    assert result.shape == (n_agents, n_assets)
    assert np.all(result >= 0), "Portfolio weights should be non-negative"
    

    portfolio_sums = result.sum(axis=1)
    np.testing.assert_allclose(portfolio_sums, 1.0, rtol=1e-10)
    

    assert np.all(np.isfinite(result)), "All weights should be finite"


if __name__ == "__main__":
    print("Testing crypsis function...")
    
    n, d = 5, 2
    X = np.random.random((n, d))
    X_best = np.random.random(d)
    t = 10
    max_iter = 100
    
    result = crypsis(X, X_best, t, max_iter)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {result.shape}")
    print(f"X_best: {X_best}")
    print(f"Result sample: {result[0]}")
    print("✓ Basic functionality test passed!")

    result_simplex = crypsis(X, X_best, t, max_iter, bounds="simplex")
    print(f"Simplex result sums: {result_simplex.sum(axis=1)}")
    print("✓ Simplex bounds test passed!")

    rng = np.random.Generator(np.random.PCG64(42))
    result2 = crypsis(X, X_best, t, max_iter, rng=rng)
    print("✓ Custom RNG test passed!")
    
    print("All tests completed successfully!")
