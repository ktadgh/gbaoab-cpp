An implementation in CUDA of the geodesic-BAOAB algorithm [DOI: 10.1098/rspa.2016.0138](https://doi.org/10.1098/rspa.2016.0138)

At the moment the implementation is limited to the hypersphere, since in CUDA I hardcoded the Jacobian at $x$ to be equal to $2x$. In order to
get a fair comparison, I did the same in the PyTorch implementation.

#### Results
```bash
CUDA: time per iteration = 4.957786798477173 ms
PyTorch: time per iteration = 46.069581508636475 ms
```