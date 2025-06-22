#include <torch/extension.h>

void rattle_hard_launcher(torch::Tensor x, torch::Tensor v, float h, int  n);

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rattle_hard", &rattle_hard_launcher, "Rattle Hard CUDA kernel");
}
