import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

path_source = "src/liberate/gpu"
ext_modules = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            os.path.join(path_source, "csprng/randint.cpp"),
            os.path.join(path_source, "csprng/randint_cuda_kernel.cu"),
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            os.path.join(path_source, "csprng/randround.cpp"),
            os.path.join(path_source, "csprng/randround_cuda_kernel.cu"),
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            os.path.join(path_source, "csprng/discrete_gaussian.cpp"),
            os.path.join(path_source, "csprng/discrete_gaussian_cuda_kernel.cu"),
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            os.path.join(path_source, "csprng/chacha20.cpp"),
            os.path.join(path_source, "csprng/chacha20_cuda_kernel.cu"),
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            os.path.join(path_source, "ntt/ntt.cpp"),
            os.path.join(path_source, "ntt/ntt_cuda_kernel.cu"),
        ],
    )
]

if __name__ == "__main__":
    setup(
        name="csprng",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext"],
        options={
            "build": {
                "build_lib": "src/liberate/csprng",
            }
        },
    )

    setup(
        name="ntt",
        ext_modules=ext_modules_ntt,
        script_args=["build_ext"],
        cmdclass={"build_ext": BuildExtension},
        options={
            "build": {
                "build_lib": "src/liberate/ntt",
            }
        },
    )
