An implementation in CUDA of the geodesic-BAOAB algorithm [DOI: 10.1098/rspa.2016.0138](https://doi.org/10.1098/rspa.2016.0138)

At the moment the implementation is limited to the hypersphere, since in CUDA I hardcoded the Jacobian at $x$ to be equal to $2x$. In order to
get a fair comparison, I did the same in the PyTorch implementation.

#### Results - RATTLE Step
```bash
CUDA: time per iteration = 0.19701995849609374 ms
PyTorch: time per iteration = 1.85897216796875 ms
```

#### To Do
##### g-BAOAB integrator
g-BAOAB should be simple to improve in the same way as rattle, and likely the performance gain will be much larger overall, since RATTLE is only run a handful of times per g-BAOAB step

##### kernel improvements
Although almost 10x faster than the PyTorch loop, the RATTLE kernel still has a lot of redundancy mainly in the conversion between batched arrays of pointers and flattened arrays. Will need to either convert the elementwise functions or the GEMM functions to avoid the conversion

##### General manifolds
Should add a function to find the Jacobian of a general manifold and pass it directly into the kernel as a function. The Jacobian can be found symbolically using SymPy