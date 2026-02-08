"""
Universal Manifold Engine - Reconstruction Module
Implements Phase-Space Reconstruction using Takens' Embedding Theorem
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional


class ManifoldReconstructor:
    """
    Phase-Space Reconstruction using optimal time delay and embedding dimension.
    
    Based on Takens' Embedding Theorem: Maps 1D time-series into m-dimensional 
    state-space preserving topological properties of the original dynamical system.
    """
    
    def __init__(self, signal: np.ndarray):
        """
        Initialize reconstructor with a scalar time-series.
        
        Args:
            signal: 1D numpy array of time-series data
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D array")
        self.signal = signal
        self.tau: Optional[int] = None
        self.m: Optional[int] = None
        self.attractor: Optional[np.ndarray] = None
    
    def mutual_information(self, max_lag: int = 50, bins: int = 16) -> np.ndarray:
        """
        Calculate mutual information for time delays to find optimal tau.
        
        Uses histogram-based estimation: I(X;Y) = H(X) + H(Y) - H(X,Y)
        where H is Shannon entropy.
        
        Args:
            max_lag: Maximum time delay to test
            bins: Number of histogram bins for probability estimation
            
        Returns:
            Array of mutual information values for each lag
        """
        signal = self.signal
        mi = np.zeros(max_lag)
        
        for lag in range(1, max_lag):
            # Create delayed versions
            x = signal[:-lag]
            y = signal[lag:]
            
            # Compute 2D histogram for joint distribution
            hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
            hist_2d = hist_2d / np.sum(hist_2d)  # Normalize
            
            # Compute marginal distributions
            px = np.sum(hist_2d, axis=1)
            py = np.sum(hist_2d, axis=0)
            
            # Calculate mutual information
            # I(X;Y) = sum_ij p(x_i, y_j) * log(p(x_i, y_j) / (p(x_i) * p(y_j)))
            pxy = hist_2d.flatten()
            px_py = np.outer(px, py).flatten()
            
            # Avoid log(0)
            mask = (pxy > 0) & (px_py > 0)
            mi[lag] = np.sum(pxy[mask] * np.log(pxy[mask] / px_py[mask]))
        
        return mi
    
    def optimal_time_delay(self, max_lag: int = 50) -> int:
        """
        Find optimal time delay using first minimum of mutual information.
        
        Args:
            max_lag: Maximum lag to search
            
        Returns:
            Optimal time delay tau
        """
        mi = self.mutual_information(max_lag)
        
        # Find first local minimum
        for i in range(1, len(mi) - 1):
            if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
                self.tau = i
                return i
        
        # If no minimum found, use lag with minimum MI
        self.tau = np.argmin(mi[1:]) + 1
        return self.tau
    
    def false_nearest_neighbors(self, tau: int, max_dim: int = 10, 
                                rtol: float = 15.0, atol: float = 2.0) -> np.ndarray:
        """
        Calculate False Nearest Neighbors ratio for each embedding dimension.
        
        A point is a "false neighbor" if distance increases significantly when 
        embedding dimension increases, indicating manifold self-intersection.
        
        Args:
            tau: Time delay
            max_dim: Maximum embedding dimension to test
            rtol: Relative tolerance threshold (typically 10-50)
            atol: Absolute tolerance for attractor size scaling
            
        Returns:
            Array of FNN ratios for each dimension
        """
        signal = self.signal
        fnn_ratios = np.zeros(max_dim)
        
        # Standard deviation for attractor size reference
        Ra = np.std(signal)
        
        for dim in range(1, max_dim):
            # Embed in dimension m
            N = len(signal) - dim * tau
            if N < 10:
                break
                
            embedded = self._embed(signal, dim, tau)
            
            # Need one more dimension for testing
            embedded_plus = self._embed(signal, dim + 1, tau)
            
            # Build KD-tree for efficient nearest neighbor search
            tree = KDTree(embedded)
            
            false_neighbors = 0
            total_neighbors = 0
            
            # For each point, find nearest neighbor
            for i in range(len(embedded)):
                # Query for 2 nearest (point itself + nearest neighbor)
                distances, indices = tree.query(embedded[i], k=2)
                
                if len(distances) < 2:
                    continue
                
                # Nearest neighbor (excluding self)
                nn_dist = distances[1]
                nn_idx = indices[1]
                
                if nn_dist == 0:
                    continue
                
                # Check if still neighbors in higher dimension
                dist_plus = np.linalg.norm(embedded_plus[i] - embedded_plus[nn_idx])
                
                # Criterion 1: Relative distance increase
                ratio = (dist_plus - nn_dist) / nn_dist
                
                # Criterion 2: Absolute distance relative to attractor size
                if ratio > rtol or dist_plus / Ra > atol:
                    false_neighbors += 1
                
                total_neighbors += 1
            
            if total_neighbors > 0:
                fnn_ratios[dim] = false_neighbors / total_neighbors
        
        return fnn_ratios
    
    def optimal_embedding_dimension(self, tau: Optional[int] = None, 
                                   max_dim: int = 10, threshold: float = 0.05) -> int:
        """
        Determine optimal embedding dimension using FNN algorithm.
        
        Args:
            tau: Time delay (uses self.tau if None)
            max_dim: Maximum dimension to test
            threshold: FNN ratio threshold for convergence
            
        Returns:
            Optimal embedding dimension m
        """
        if tau is None:
            if self.tau is None:
                tau = self.optimal_time_delay()
            else:
                tau = self.tau
        
        fnn = self.false_nearest_neighbors(tau, max_dim)
        
        # Find first dimension where FNN ratio drops below threshold
        for dim in range(1, max_dim):
            if fnn[dim] < threshold:
                self.m = dim + 1  # +1 because we need the dimension, not index
                return self.m
        
        # If no convergence, use dimension with minimum FNN
        self.m = np.argmin(fnn[1:]) + 2
        return self.m
    
    def _embed(self, signal: np.ndarray, dim: int, tau: int) -> np.ndarray:
        """
        Create time-delay embedding.
        
        Args:
            signal: Input time-series
            dim: Embedding dimension
            tau: Time delay
            
        Returns:
            Embedded vectors of shape (N - (dim-1)*tau, dim)
        """
        N = len(signal) - (dim - 1) * tau
        embedded = np.zeros((N, dim))
        
        for i in range(dim):
            embedded[:, i] = signal[i * tau : i * tau + N]
        
        return embedded
    
    def reconstruct(self, tau: Optional[int] = None, 
                   m: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct phase-space attractor using Takens' embedding.
        
        Args:
            tau: Time delay (auto-computed if None)
            m: Embedding dimension (auto-computed if None)
            
        Returns:
            Phase-space attractor of shape (N, m)
        """
        if tau is None:
            if self.tau is None:
                tau = self.optimal_time_delay()
            else:
                tau = self.tau
        else:
            self.tau = tau
        
        if m is None:
            if self.m is None:
                m = self.optimal_embedding_dimension(tau)
            else:
                m = self.m
        else:
            self.m = m
        
        self.attractor = self._embed(self.signal, m, tau)
        return self.attractor
    
    def get_parameters(self) -> dict:
        """
        Get reconstruction parameters.
        
        Returns:
            Dictionary with tau, m, and attractor shape
        """
        return {
            'tau': self.tau,
            'm': self.m,
            'attractor_shape': self.attractor.shape if self.attractor is not None else None
        }
