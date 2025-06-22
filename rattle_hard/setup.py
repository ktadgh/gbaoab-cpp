import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

TORCH_CUDA_LIB_DIR = os.path.join(torch.__path__[0], 'lib')  # PyTorch's CUDA .so files
TORCH_CUDA_INCLUDE_DIR = os.path.join(CUDA_HOME, 'include') if CUDA_HOME else None

setup(
    name='rattle_hard_cuda',
    ext_modules=[
        CUDAExtension(
            name='rattle_hard_cuda',
            sources=['rattle_hard.cpp', 'rattle_hard_kernel.cu'],
            libraries=['cublas', 'cusolver'],
            library_dirs=[TORCH_CUDA_LIB_DIR],
            include_dirs=[TORCH_CUDA_INCLUDE_DIR] if TORCH_CUDA_INCLUDE_DIR else [],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)