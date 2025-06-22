#include <torch/extension.h>

// Declare your function (assuming you refactor rattleHard to work with torch::Tensor)
void rattle_hard_launcher(torch::Tensor x, torch::Tensor v, float h);

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rattle_hard", &rattle_hard_launcher, "Rattle Hard CUDA kernel");
}
