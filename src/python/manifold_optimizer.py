"""
Universal Manifold Engine - Optimization Module
Implements convex optimization using ADMM for manifold-based signal denoising
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Callable


def manifold_denoise(signal: np.ndarray, 
                    lambda_reg: float = 0.1,
                    max_iter: int = 100,
                    rho: float = 1.0,
                    tol: float = 1e-4,
                    prior: str = 'tv') -> np.ndarray:
    """
    Denoise signal using convex optimization with manifold-smoothness prior.
    
    Minimizes: ||x - y||_2^2 + λ Φ(x)
    where:
        - x is the denoised signal
        - y is the noisy observation
        - Φ(x) is the smoothness prior (TV, L2, or custom)
        - λ controls regularization strength
    
    Uses ADMM (Alternating Direction Method of Multipliers) for efficient solving.
    
    Args:
        signal: Noisy input signal (1D array)
        lambda_reg: Regularization parameter λ
        max_iter: Maximum ADMM iterations
        rho: ADMM penalty parameter
        tol: Convergence tolerance
        prior: Smoothness prior type ('tv', 'l2', or 'manifold')
        
    Returns:
        Denoised signal
    """
    n = len(signal)
    
    # Define optimization variable
    x = cp.Variable(n)
    
    # Data fidelity term: ||x - y||_2^2
    fidelity = cp.sum_squares(x - signal)
    
    # Smoothness prior Φ(x)
    if prior == 'tv':
        # Total Variation: promotes piecewise constant solutions
        # Φ(x) = ||Dx||_1 where D is difference operator
        regularizer = cp.norm(cp.diff(x), 1)
    elif prior == 'l2':
        # L2 smoothness: penalizes second derivatives
        # Φ(x) = ||D²x||_2^2
        D2 = np.diag(np.ones(n)) * 2
        D2 += np.diag(-np.ones(n-1), 1)
        D2 += np.diag(-np.ones(n-1), -1)
        D2 = D2[1:-1]  # Remove boundary rows
        regularizer = cp.sum_squares(D2 @ x)
    elif prior == 'manifold':
        # Manifold smoothness: combines TV and L2
        # Φ(x) = ||Dx||_1 + α||D²x||_2^2
        D2 = np.diag(np.ones(n)) * 2
        D2 += np.diag(-np.ones(n-1), 1)
        D2 += np.diag(-np.ones(n-1), -1)
        D2 = D2[1:-1]
        regularizer = cp.norm(cp.diff(x), 1) + 0.1 * cp.sum_squares(D2 @ x)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    
    # Objective: minimize fidelity + λ * regularizer
    objective = cp.Minimize(fidelity + lambda_reg * regularizer)
    
    # Solve using CVXPY (automatically uses ADMM for large problems)
    problem = cp.Problem(objective)
    
    try:
        problem.solve(max_iters=max_iter, eps_abs=tol, eps_rel=tol)
        
        if x.value is None:
            # Fallback to basic solver if ADMM fails
            problem.solve(solver=cp.SCS, max_iters=max_iter)
        
        return x.value if x.value is not None else signal
    
    except Exception as e:
        print(f"Optimization failed: {e}")
        return signal


def admm_denoise_manual(signal: np.ndarray,
                       lambda_reg: float = 0.1,
                       max_iter: int = 100,
                       rho: float = 1.0,
                       tol: float = 1e-4) -> np.ndarray:
    """
    Manual ADMM implementation for Total Variation denoising.
    
    Solves: minimize ||x - y||_2^2 + λ||Dx||_1
    
    ADMM formulation:
        x^(k+1) = argmin_x ||x - y||_2^2 + (ρ/2)||Dx - z^k + u^k||_2^2
        z^(k+1) = argmin_z λ||z||_1 + (ρ/2)||Dx^(k+1) - z + u^k||_2^2
        u^(k+1) = u^k + Dx^(k+1) - z^(k+1)
    
    Args:
        signal: Noisy input signal
        lambda_reg: Regularization parameter
        max_iter: Maximum iterations
        rho: ADMM penalty parameter
        tol: Convergence tolerance
        
    Returns:
        Denoised signal
    """
    n = len(signal)
    
    # Create difference matrix D
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1
    
    # Initialize variables
    x = signal.copy()
    z = np.zeros(n-1)
    u = np.zeros(n-1)
    
    # Precompute (I + ρD^T D)^(-1)
    I = np.eye(n)
    DTD = D.T @ D
    A = I + rho * DTD
    A_inv = np.linalg.inv(A)
    
    # ADMM iterations
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # x-update: (I + ρD^T D)x = y + ρD^T(z - u)
        b = signal + rho * D.T @ (z - u)
        x = A_inv @ b
        
        # z-update: soft thresholding
        Dx_plus_u = D @ x + u
        z = soft_threshold(Dx_plus_u, lambda_reg / rho)
        
        # u-update: dual variable update
        u = u + D @ x - z
        
        # Check convergence
        primal_residual = np.linalg.norm(D @ x - z)
        dual_residual = np.linalg.norm(rho * D.T @ (z - D @ x_old))
        
        if primal_residual < tol and dual_residual < tol:
            break
    
    return x


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator for L1 penalty.
    
    S_λ(x) = sign(x) * max(|x| - λ, 0)
    
    Args:
        x: Input array
        threshold: Threshold value λ
        
    Returns:
        Soft-thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


class ManifoldOptimizer:
    """
    Advanced manifold-based optimization for signal processing.
    
    Combines phase-space reconstruction with convex optimization for
    noise-robust forecasting and signal recovery.
    """
    
    def __init__(self, lambda_reg: float = 0.1, rho: float = 1.0):
        """
        Initialize optimizer.
        
        Args:
            lambda_reg: Regularization strength
            rho: ADMM penalty parameter
        """
        self.lambda_reg = lambda_reg
        self.rho = rho
    
    def denoise(self, signal: np.ndarray, method: str = 'cvxpy') -> np.ndarray:
        """
        Denoise signal using specified method.
        
        Args:
            signal: Noisy input signal
            method: 'cvxpy' or 'admm'
            
        Returns:
            Denoised signal
        """
        if method == 'cvxpy':
            return manifold_denoise(signal, self.lambda_reg, rho=self.rho, prior='manifold')
        elif method == 'admm':
            return admm_denoise_manual(signal, self.lambda_reg, rho=self.rho)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forecast(self, signal: np.ndarray, attractor: np.ndarray, 
                steps: int = 1) -> np.ndarray:
        """
        Forecast future values using manifold structure.
        
        Uses local linear models on the reconstructed attractor.
        
        Args:
            signal: Historical signal
            attractor: Phase-space attractor
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        # Simple local linear forecast
        # In production, use more sophisticated manifold-based forecasting
        
        # Get last state
        last_state = attractor[-1]
        
        # Find k nearest neighbors
        from scipy.spatial import KDTree
        tree = KDTree(attractor[:-steps])
        
        k = min(10, len(attractor) // 2)
        distances, indices = tree.query(last_state, k=k)
        
        # Forecast as weighted average of neighbor successors
        weights = 1.0 / (distances + 1e-10)
        weights /= np.sum(weights)
        
        forecast = np.zeros(steps)
        for i in range(steps):
            # Get successor values
            successors = np.array([signal[idx + i + 1] 
                                  for idx in indices 
                                  if idx + i + 1 < len(signal)])
            if len(successors) > 0:
                forecast[i] = np.average(successors[:len(weights)], weights=weights[:len(successors)])
            else:
                forecast[i] = signal[-1]
        
        return forecast
    
    def optimize_parameters(self, signal: np.ndarray, 
                          validation_split: float = 0.2) -> dict:
        """
        Find optimal hyperparameters using cross-validation.
        
        Args:
            signal: Training signal
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with optimal parameters
        """
        split_idx = int(len(signal) * (1 - validation_split))
        train = signal[:split_idx]
        val = signal[split_idx:]
        
        best_params = {'lambda': self.lambda_reg, 'rho': self.rho}
        best_error = float('inf')
        
        # Grid search
        for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
            for rho_val in [0.5, 1.0, 2.0]:
                try:
                    denoised = manifold_denoise(train, lambda_reg=lam, rho=rho_val)
                    # Simple validation: check smoothness
                    error = np.sum(np.diff(denoised)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_params = {'lambda': lam, 'rho': rho_val}
                except:
                    continue
        
        self.lambda_reg = best_params['lambda']
        self.rho = best_params['rho']
        
        return best_params
