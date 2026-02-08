#!/usr/bin/env python3
"""
Integration test for Universal Manifold Engine
Tests all components together
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

import numpy as np
from manifold_reconstructor import ManifoldReconstructor
from manifold_optimizer import manifold_denoise, admm_denoise_manual, ManifoldOptimizer


def test_reconstruction():
    """Test phase-space reconstruction"""
    print("Testing Reconstruction...")
    
    # Generate Lorenz-like chaotic signal
    signal = np.cumsum(np.random.randn(500)) * 0.1
    
    reconstructor = ManifoldReconstructor(signal)
    tau = reconstructor.optimal_time_delay(max_lag=30)
    m = reconstructor.optimal_embedding_dimension(tau=tau, max_dim=6)
    attractor = reconstructor.reconstruct(tau=tau, m=m)
    
    assert tau > 0, "Time delay should be positive"
    assert m > 0, "Embedding dimension should be positive"
    assert attractor.shape[0] > 0, "Attractor should have points"
    assert attractor.shape[1] == m, "Attractor should have m dimensions"
    
    print(f"  ✓ τ={tau}, m={m}, attractor shape={attractor.shape}")
    return True


def test_denoising():
    """Test signal denoising"""
    print("Testing Denoising...")
    
    # Clean signal
    t = np.linspace(0, 10, 200)
    clean = np.sin(2*np.pi*t)
    noisy = clean + np.random.normal(0, 0.3, len(clean))
    
    # Test CVXPY
    denoised_cvxpy = manifold_denoise(noisy, lambda_reg=0.2, prior='tv')
    error_cvxpy = np.mean((denoised_cvxpy - clean)**2)
    
    # Test ADMM
    denoised_admm = admm_denoise_manual(noisy, lambda_reg=0.2)
    error_admm = np.mean((denoised_admm - clean)**2)
    
    # Should improve over noisy
    error_noisy = np.mean((noisy - clean)**2)
    
    assert error_cvxpy < error_noisy, "CVXPY should improve signal"
    assert error_admm < error_noisy, "ADMM should improve signal"
    
    improvement_cvxpy = (1 - error_cvxpy/error_noisy) * 100
    improvement_admm = (1 - error_admm/error_noisy) * 100
    
    print(f"  ✓ CVXPY improvement: {improvement_cvxpy:.1f}%")
    print(f"  ✓ ADMM improvement: {improvement_admm:.1f}%")
    return True


def test_optimizer():
    """Test ManifoldOptimizer class"""
    print("Testing ManifoldOptimizer...")
    
    signal = np.sin(np.linspace(0, 20, 300)) + np.random.normal(0, 0.2, 300)
    
    optimizer = ManifoldOptimizer(lambda_reg=0.1, rho=1.0)
    denoised = optimizer.denoise(signal, method='admm')
    
    assert len(denoised) == len(signal), "Output length should match input"
    assert np.std(np.diff(denoised)) < np.std(np.diff(signal)), "Should be smoother"
    
    print(f"  ✓ Denoised signal, smoothness improved")
    return True


def test_forecasting():
    """Test manifold-based forecasting"""
    print("Testing Forecasting...")
    
    signal = np.cumsum(np.random.randn(400)) * 0.1
    
    reconstructor = ManifoldReconstructor(signal)
    tau = reconstructor.optimal_time_delay(max_lag=20)
    attractor = reconstructor.reconstruct(tau=tau, m=3)
    
    optimizer = ManifoldOptimizer()
    forecast = optimizer.forecast(signal, attractor, steps=5)
    
    assert len(forecast) == 5, "Should forecast 5 steps"
    assert not np.any(np.isnan(forecast)), "Forecast should not contain NaN"
    
    print(f"  ✓ Forecast: {forecast[:3]}...")
    return True


def test_end_to_end():
    """Test complete pipeline"""
    print("Testing End-to-End Pipeline...")
    
    # Generate signal
    t = np.linspace(0, 30, 600)
    signal = np.sin(t) + 0.5*np.sin(3*t) + np.random.normal(0, 0.3, len(t))
    
    # Step 1: Denoise
    denoised = manifold_denoise(signal, lambda_reg=0.1, prior='manifold')
    
    # Step 2: Reconstruct
    reconstructor = ManifoldReconstructor(denoised)
    tau = reconstructor.optimal_time_delay()
    m = reconstructor.optimal_embedding_dimension(tau=tau)
    attractor = reconstructor.reconstruct()
    
    # Step 3: Forecast
    optimizer = ManifoldOptimizer()
    forecast = optimizer.forecast(denoised, attractor, steps=10)
    
    print(f"  ✓ Pipeline: denoise → reconstruct(τ={tau}, m={m}) → forecast")
    print(f"  ✓ Complete!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Universal Manifold Engine - Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_reconstruction,
        test_denoising,
        test_optimizer,
        test_forecasting,
        test_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
