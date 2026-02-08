/*
 * Universal Manifold Engine - C++ Hardware Implementation
 * Optimized for ARM Cortex-M and FPGA with fixed-point arithmetic
 */

#ifndef MANIFOLD_ENGINE_HPP
#define MANIFOLD_ENGINE_HPP

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace manifold {

// Fixed-point configuration
constexpr int32_t FIXED_POINT_SHIFT = 16;  // Q16.16 format
constexpr int32_t FIXED_POINT_ONE = 1 << FIXED_POINT_SHIFT;
constexpr int32_t FIXED_POINT_HALF = FIXED_POINT_ONE / 2;

/**
 * Convert floating point to fixed point
 */
inline int32_t to_fixed(double x) {
    return static_cast<int32_t>(x * FIXED_POINT_ONE + (x >= 0 ? 0.5 : -0.5));
}

/**
 * Convert fixed point to floating point
 */
inline double to_float(int32_t x) {
    return static_cast<double>(x) / FIXED_POINT_ONE;
}

/**
 * Fixed-point multiplication
 */
inline int32_t fixed_mul(int32_t a, int32_t b) {
    int64_t product = static_cast<int64_t>(a) * b;
    return static_cast<int32_t>((product + FIXED_POINT_HALF) >> FIXED_POINT_SHIFT);
}

/**
 * Fixed-point division
 * Note: Returns extreme values on division by zero for embedded systems
 * where exception handling may not be available. Check divisor before calling.
 */
inline int32_t fixed_div(int32_t a, int32_t b) {
    if (b == 0) return (a >= 0) ? std::numeric_limits<int32_t>::max() 
                                 : std::numeric_limits<int32_t>::min();
    int64_t quotient = (static_cast<int64_t>(a) << FIXED_POINT_SHIFT) / b;
    return static_cast<int32_t>(quotient);
}

/**
 * Fixed-point square root (Newton-Raphson)
 */
inline int32_t fixed_sqrt(int32_t x) {
    if (x <= 0) return 0;
    
    // Initial guess
    int32_t result = x;
    int32_t prev = 0;
    
    // Newton iterations: x_new = (x_old + n/x_old) / 2
    for (int i = 0; i < 10; ++i) {
        prev = result;
        result = (result + fixed_div(x, result)) >> 1;
        if (result == prev) break;
    }
    
    return result;
}

/**
 * Memory-efficient vector operations
 */
template<typename T>
class Vector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    Vector() : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit Vector(size_t n) : size_(n), capacity_(n) {
        data_ = new T[n];
        for (size_t i = 0; i < n; ++i) data_[i] = 0;
    }
    
    ~Vector() { delete[] data_; }
    
    // Copy constructor
    Vector(const Vector& other) : size_(other.size_), capacity_(other.capacity_) {
        data_ = new T[capacity_];
        for (size_t i = 0; i < size_; ++i) data_[i] = other.data_[i];
    }
    
    // Copy assignment operator (Rule of Three)
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = new T[capacity_];
            for (size_t i = 0; i < size_; ++i) data_[i] = other.data_[i];
        }
        return *this;
    }
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
    size_t size() const { return size_; }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            capacity_ = capacity_ == 0 ? 1 : capacity_ * 2;
            T* new_data = new T[capacity_];
            for (size_t i = 0; i < size_; ++i) new_data[i] = data_[i];
            delete[] data_;
            data_ = new_data;
        }
        data_[size_++] = value;
    }
};

/**
 * Phase-Space Reconstructor (Hardware-Optimized)
 */
class ManifoldReconstructor {
private:
    const int32_t* signal_;
    size_t signal_len_;
    int32_t tau_;
    int32_t m_;
    
public:
    ManifoldReconstructor(const int32_t* signal, size_t len) 
        : signal_(signal), signal_len_(len), tau_(0), m_(0) {}
    
    /**
     * Calculate mutual information for time delay estimation
     * Uses fixed-point arithmetic for embedded systems
     */
    int32_t estimate_time_delay(int32_t max_lag = 50) {
        if (max_lag <= 0 || max_lag >= static_cast<int32_t>(signal_len_)) {
            max_lag = signal_len_ / 10;
        }
        
        int32_t* mi = new int32_t[max_lag];
        
        for (int32_t lag = 1; lag < max_lag; ++lag) {
            mi[lag] = compute_mutual_info(lag);
        }
        
        // Find first minimum
        tau_ = 1;
        for (int32_t i = 2; i < max_lag - 1; ++i) {
            if (mi[i] < mi[i-1] && mi[i] < mi[i+1]) {
                tau_ = i;
                break;
            }
        }
        
        delete[] mi;
        return tau_;
    }
    
    /**
     * Estimate embedding dimension using False Nearest Neighbors
     */
    int32_t estimate_dimension(int32_t tau, int32_t max_dim = 10) {
        if (tau <= 0) tau = estimate_time_delay();
        
        tau_ = tau;
        int32_t best_dim = 2;
        int32_t min_fnn = FIXED_POINT_ONE;  // 100%
        
        for (int32_t dim = 1; dim < max_dim; ++dim) {
            int32_t fnn_ratio = compute_fnn_ratio(dim, tau);
            
            // If FNN ratio drops below 5%, use this dimension
            if (fnn_ratio < (FIXED_POINT_ONE / 20)) {
                best_dim = dim + 1;
                break;
            }
            
            if (fnn_ratio < min_fnn) {
                min_fnn = fnn_ratio;
                best_dim = dim + 1;
            }
        }
        
        m_ = best_dim;
        return best_dim;
    }
    
    /**
     * Reconstruct phase-space attractor
     */
    void reconstruct(int32_t* output, size_t* output_len) {
        if (tau_ <= 0) estimate_time_delay();
        if (m_ <= 0) estimate_dimension(tau_);
        
        size_t n_points = signal_len_ - (m_ - 1) * tau_;
        *output_len = n_points * m_;
        
        // Embed signal: output[i*m + j] = signal[i + j*tau]
        for (size_t i = 0; i < n_points; ++i) {
            for (int32_t j = 0; j < m_; ++j) {
                output[i * m_ + j] = signal_[i + j * tau_];
            }
        }
    }
    
    int32_t get_tau() const { return tau_; }
    int32_t get_dimension() const { return m_; }

private:
    /**
     * Compute mutual information for given lag (simplified histogram method)
     */
    int32_t compute_mutual_info(int32_t lag) {
        constexpr int32_t bins = 16;
        
        // Create histograms (simplified for embedded systems)
        int32_t hist_2d[bins][bins] = {0};
        
        // Find min/max for binning
        int32_t min_val = signal_[0];
        int32_t max_val = signal_[0];
        for (size_t i = 0; i < signal_len_; ++i) {
            if (signal_[i] < min_val) min_val = signal_[i];
            if (signal_[i] > max_val) max_val = signal_[i];
        }
        
        int32_t range = max_val - min_val;
        if (range == 0) return 0;
        
        // Fill 2D histogram
        int32_t count = 0;
        for (size_t i = 0; i < signal_len_ - lag; ++i) {
            int32_t bin_x = ((signal_[i] - min_val) * (bins - 1)) / range;
            int32_t bin_y = ((signal_[i + lag] - min_val) * (bins - 1)) / range;
            
            if (bin_x >= 0 && bin_x < bins && bin_y >= 0 && bin_y < bins) {
                hist_2d[bin_x][bin_y]++;
                count++;
            }
        }
        
        if (count == 0) return 0;
        
        // Compute marginals
        int32_t px[bins] = {0};
        int32_t py[bins] = {0};
        for (int32_t i = 0; i < bins; ++i) {
            for (int32_t j = 0; j < bins; ++j) {
                px[i] += hist_2d[i][j];
                py[j] += hist_2d[i][j];
            }
        }
        
        // Compute MI (simplified integer arithmetic)
        int32_t mi = 0;
        for (int32_t i = 0; i < bins; ++i) {
            for (int32_t j = 0; j < bins; ++j) {
                if (hist_2d[i][j] > 0 && px[i] > 0 && py[j] > 0) {
                    // MI += p(x,y) * log(p(x,y) / (p(x)*p(y)))
                    // Simplified: use ratio as proxy
                    int32_t joint = hist_2d[i][j];
                    int32_t independent = (px[i] * py[j]) / count;
                    if (independent > 0) {
                        mi += (joint - independent);  // Simplified metric
                    }
                }
            }
        }
        
        return mi;
    }
    
    /**
     * Compute False Nearest Neighbors ratio
     */
    int32_t compute_fnn_ratio(int32_t dim, int32_t tau) {
        size_t n_points = signal_len_ - dim * tau;
        if (n_points < 10) return FIXED_POINT_ONE;
        
        int32_t false_neighbors = 0;
        int32_t total_neighbors = 0;
        
        // Tolerance thresholds (fixed-point)
        constexpr int32_t rtol = 15 * FIXED_POINT_ONE;
        
        // Sample subset for efficiency
        size_t sample_step = n_points > 100 ? n_points / 100 : 1;
        
        for (size_t i = 0; i < n_points; i += sample_step) {
            // Find nearest neighbor (simplified linear search)
            int32_t min_dist = std::numeric_limits<int32_t>::max();
            size_t nn_idx = 0;
            
            for (size_t j = 0; j < n_points; ++j) {
                if (i == j) continue;
                
                // Compute distance in dim-dimensional space
                int64_t dist_sq = 0;
                for (int32_t k = 0; k < dim; ++k) {
                    int32_t diff = signal_[i + k * tau] - signal_[j + k * tau];
                    dist_sq += static_cast<int64_t>(diff) * diff;
                }
                
                int32_t dist = fixed_sqrt(static_cast<int32_t>(dist_sq >> FIXED_POINT_SHIFT));
                if (dist < min_dist && dist > 0) {
                    min_dist = dist;
                    nn_idx = j;
                }
            }
            
            if (min_dist == 0 || min_dist == std::numeric_limits<int32_t>::max()) {
                continue;
            }
            
            // Check in (dim+1)-dimensional space
            int64_t dist_plus_sq = 0;
            for (int32_t k = 0; k <= dim; ++k) {
                if (i + k * tau < signal_len_ && nn_idx + k * tau < signal_len_) {
                    int32_t diff = signal_[i + k * tau] - signal_[nn_idx + k * tau];
                    dist_plus_sq += static_cast<int64_t>(diff) * diff;
                }
            }
            
            int32_t dist_plus = fixed_sqrt(static_cast<int32_t>(dist_plus_sq >> FIXED_POINT_SHIFT));
            
            // Check if false neighbor
            int32_t ratio = fixed_div(dist_plus - min_dist, min_dist);
            if (ratio > rtol) {
                false_neighbors++;
            }
            
            total_neighbors++;
        }
        
        if (total_neighbors == 0) return FIXED_POINT_ONE;
        
        return fixed_div(to_fixed(false_neighbors), to_fixed(total_neighbors));
    }
};

/**
 * ADMM-based denoising optimizer (Hardware-Optimized)
 */
class ManifoldOptimizer {
private:
    int32_t lambda_;  // Regularization parameter (fixed-point)
    int32_t rho_;     // ADMM penalty parameter (fixed-point)
    
public:
    ManifoldOptimizer(int32_t lambda = FIXED_POINT_ONE / 10, 
                      int32_t rho = FIXED_POINT_ONE)
        : lambda_(lambda), rho_(rho) {}
    
    /**
     * Denoise signal using ADMM with Total Variation prior
     */
    void denoise(const int32_t* input, int32_t* output, size_t len, int32_t max_iter = 50) {
        // Initialize variables
        int32_t* x = new int32_t[len];
        int32_t* z = new int32_t[len - 1];
        int32_t* u = new int32_t[len - 1];
        
        for (size_t i = 0; i < len; ++i) x[i] = input[i];
        for (size_t i = 0; i < len - 1; ++i) {
            z[i] = 0;
            u[i] = 0;
        }
        
        // ADMM iterations
        for (int32_t iter = 0; iter < max_iter; ++iter) {
            // x-update: solve (I + ρD^T D)x = y + ρD^T(z - u)
            x_update(input, x, z, u, len);
            
            // z-update: soft thresholding
            z_update(x, z, u, len);
            
            // u-update: dual variable
            u_update(x, z, u, len);
        }
        
        // Copy result
        for (size_t i = 0; i < len; ++i) {
            output[i] = x[i];
        }
        
        delete[] x;
        delete[] z;
        delete[] u;
    }

private:
    void x_update(const int32_t* y, int32_t* x, const int32_t* z, const int32_t* u, size_t len) {
        // Simplified Jacobi iteration for (I + ρD^T D)x = b
        int32_t* x_new = new int32_t[len];
        
        for (int iter = 0; iter < 5; ++iter) {
            for (size_t i = 0; i < len; ++i) {
                int32_t b = y[i];
                
                // Add ρD^T(z - u) contribution
                if (i > 0) {
                    b += fixed_mul(rho_, z[i-1] - u[i-1]);
                }
                if (i < len - 1) {
                    b -= fixed_mul(rho_, z[i] - u[i]);
                }
                
                // Diagonal dominance: (1 + 2ρ)
                int32_t diag = FIXED_POINT_ONE + (rho_ << 1);
                x_new[i] = fixed_div(b, diag);
            }
            
            for (size_t i = 0; i < len; ++i) x[i] = x_new[i];
        }
        
        delete[] x_new;
    }
    
    void z_update(const int32_t* x, int32_t* z, const int32_t* u, size_t len) {
        int32_t threshold = fixed_div(lambda_, rho_);
        
        for (size_t i = 0; i < len - 1; ++i) {
            int32_t diff = x[i+1] - x[i];
            int32_t v = diff + u[i];
            
            // Soft thresholding: sign(v) * max(|v| - threshold, 0)
            int32_t abs_v = v >= 0 ? v : -v;
            if (abs_v > threshold) {
                z[i] = v >= 0 ? (abs_v - threshold) : -(abs_v - threshold);
            } else {
                z[i] = 0;
            }
        }
    }
    
    void u_update(const int32_t* x, const int32_t* z, int32_t* u, size_t len) {
        for (size_t i = 0; i < len - 1; ++i) {
            int32_t diff = x[i+1] - x[i];
            u[i] = u[i] + diff - z[i];
        }
    }
};

} // namespace manifold

#endif // MANIFOLD_ENGINE_HPP
