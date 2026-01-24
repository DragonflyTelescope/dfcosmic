// Fast median filter implementation using C++
// This would be compiled as a PyTorch extension

#include <torch/extension.h>
#include <vector>
#include <algorithm>

// Helper function to compute median of a vector
template<typename scalar_t>
scalar_t compute_median(std::vector<scalar_t>& values) {
    size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n/2, values.end());
    return values[n/2];
}

// CPU implementation
torch::Tensor median_filter_cpu(
    torch::Tensor input,
    int64_t kernel_size
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(kernel_size % 2 == 1, "Kernel size must be odd");
    
    auto input_a = input.accessor<float, 2>();
    int64_t H = input.size(0);
    int64_t W = input.size(1);
    int64_t pad = kernel_size / 2;
    
    auto output = torch::zeros_like(input);
    auto output_a = output.accessor<float, 2>();
    
    // Pre-allocate vector for window values
    std::vector<float> window;
    window.reserve(kernel_size * kernel_size);
    
    // Parallel loop over pixels
    #pragma omp parallel for collapse(2) private(window)
    for (int64_t i = 0; i < H; i++) {
        for (int64_t j = 0; j < W; j++) {
            window.clear();
            
            // Gather window values with boundary handling (replicate mode)
            for (int64_t di = -pad; di <= pad; di++) {
                for (int64_t dj = -pad; dj <= pad; dj++) {
                    int64_t ii = std::max(int64_t(0), std::min(H-1, i + di));
                    int64_t jj = std::max(int64_t(0), std::min(W-1, j + dj));
                    window.push_back(input_a[ii][jj]);
                }
            }
            
            // Compute median using partial sort (faster than full sort)
            output_a[i][j] = compute_median(window);
        }
    }
    
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("median_filter_cpu", &median_filter_cpu, "Fast median filter (CPU)");
}
