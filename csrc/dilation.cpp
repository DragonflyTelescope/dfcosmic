// Fast grayscale dilation implementation using C++
// This would be compiled as a PyTorch extension

#include <torch/extension.h>
#include <vector>

torch::Tensor dilation_cpu(
    torch::Tensor input,
    int64_t k_h,
    int64_t k_w,
    int64_t origin_y,
    int64_t origin_x,
    float border_value
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(k_h > 0 && k_w > 0, "Kernel size must be positive");

    auto input_a = input.accessor<float, 2>();
    int64_t H = input.size(0);
    int64_t W = input.size(1);

    auto output = torch::zeros_like(input);
    auto output_a = output.accessor<float, 2>();

    int64_t center_y = k_h / 2;
    int64_t center_x = k_w / 2;

    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < H; i++) {
        for (int64_t j = 0; j < W; j++) {
            float max_val = border_value;
            for (int64_t dy = 0; dy < k_h; dy++) {
                int64_t yy = i + dy - center_y - origin_y;
                if (yy < 0 || yy >= H) {
                    for (int64_t dx = 0; dx < k_w; dx++) {
                        if (border_value > max_val) {
                            max_val = border_value;
                        }
                    }
                    continue;
                }
                for (int64_t dx = 0; dx < k_w; dx++) {
                    int64_t xx = j + dx - center_x - origin_x;
                    float val = border_value;
                    if (xx >= 0 && xx < W) {
                        val = input_a[yy][xx];
                    }
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            output_a[i][j] = max_val;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dilation_cpu", &dilation_cpu, "Fast dilation (CPU)");
}
