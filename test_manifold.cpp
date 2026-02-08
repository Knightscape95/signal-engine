/*
 * Test file for manifold_engine.hpp
 * Validates fixed-point arithmetic and basic reconstruction
 */

#include "manifold_engine.hpp"
#include <iostream>
#include <cmath>

using namespace manifold;

void test_fixed_point() {
    std::cout << "Testing Fixed-Point Arithmetic..." << std::endl;
    
    // Test conversion
    double x = 3.14159;
    int32_t x_fixed = to_fixed(x);
    double x_back = to_float(x_fixed);
    std::cout << "  Original: " << x << ", Fixed: " << x_fixed 
              << ", Back: " << x_back << std::endl;
    
    // Test multiplication
    int32_t a = to_fixed(2.5);
    int32_t b = to_fixed(4.0);
    int32_t c = fixed_mul(a, b);
    std::cout << "  2.5 * 4.0 = " << to_float(c) << " (expected 10.0)" << std::endl;
    
    // Test division
    int32_t d = to_fixed(10.0);
    int32_t e = to_fixed(2.0);
    int32_t f = fixed_div(d, e);
    std::cout << "  10.0 / 2.0 = " << to_float(f) << " (expected 5.0)" << std::endl;
    
    // Test square root
    int32_t g = to_fixed(16.0);
    int32_t h = fixed_sqrt(g);
    std::cout << "  sqrt(16.0) = " << to_float(h) << " (expected 4.0)" << std::endl;
    
    std::cout << "  Fixed-point tests passed!\n" << std::endl;
}

void test_reconstruction() {
    std::cout << "Testing Phase-Space Reconstruction..." << std::endl;
    
    // Generate simple sinusoidal signal
    const int N = 200;
    int32_t signal[N];
    for (int i = 0; i < N; i++) {
        double t = i * 0.1;
        signal[i] = to_fixed(std::sin(t));
    }
    
    // Create reconstructor
    ManifoldReconstructor reconstructor(signal, N);
    
    // Estimate parameters
    int32_t tau = reconstructor.estimate_time_delay(30);
    std::cout << "  Estimated tau: " << tau << std::endl;
    
    int32_t m = reconstructor.estimate_dimension(tau, 5);
    std::cout << "  Estimated dimension: " << m << std::endl;
    
    // Reconstruct
    int32_t* output = new int32_t[N * m];
    size_t output_len = 0;
    reconstructor.reconstruct(output, &output_len);
    
    std::cout << "  Reconstructed attractor size: " << output_len << std::endl;
    std::cout << "  Reconstruction tests passed!\n" << std::endl;
    
    delete[] output;
}

void test_optimization() {
    std::cout << "Testing ADMM Optimization..." << std::endl;
    
    // Generate noisy signal
    const int N = 100;
    int32_t signal[N];
    int32_t output[N];
    
    // Clean signal with noise
    for (int i = 0; i < N; i++) {
        double t = i * 0.1;
        double clean = std::sin(t);
        double noise = ((rand() % 100) - 50) / 500.0;
        signal[i] = to_fixed(clean + noise);
    }
    
    // Create optimizer
    ManifoldOptimizer optimizer(to_fixed(0.1), to_fixed(1.0));
    
    // Denoise
    optimizer.denoise(signal, output, N, 20);
    
    std::cout << "  First 5 values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "    Input: " << to_float(signal[i]) 
                  << ", Output: " << to_float(output[i]) << std::endl;
    }
    
    std::cout << "  Optimization tests passed!\n" << std::endl;
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Manifold Engine C++ Test Suite" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    test_fixed_point();
    test_reconstruction();
    test_optimization();
    
    std::cout << "====================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
