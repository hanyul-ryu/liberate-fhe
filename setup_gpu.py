import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

path_source = "src/liberate/gpu"


class CustomBuildExt(BuildExtension):
    def build_extension(self, ext):
        if ext.name == "ntt_cuda":
            self.build_lib = os.path.join(path_source, "ntt")
        else:
            self.build_lib = os.path.join(path_source, "csprng")

        os.environ["MAX_JOBS"] = str(os.cpu_count())
        super().build_extension(ext)


ext_modules = [
    ###################
    ####    ntt    ####
    ###################
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            os.path.join(path_source, "ntt/ntt.cpp"),
            os.path.join(path_source, "ntt/ntt_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3", "-DNDEBUG"
            ]
        }
    ),
    ###################
    ####    rng    ####
    ###################
    CUDAExtension(
        name="randint_cuda",
        sources=[
            os.path.join(path_source, "csprng/randint.cpp"),
            os.path.join(path_source, "csprng/randint_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3", "-DNDEBUG"
            ]
        }
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            os.path.join(path_source, "csprng/randround.cpp"),
            os.path.join(path_source, "csprng/randround_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3", "-DNDEBUG"
            ]
        }
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            os.path.join(path_source, "csprng/discrete_gaussian.cpp"),
            os.path.join(path_source, "csprng/discrete_gaussian_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3", "-DNDEBUG"
            ]
        }
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            os.path.join(path_source, "csprng/chacha20.cpp"),
            os.path.join(path_source, "csprng/chacha20_cuda_kernel.cu"),
        ],
        extra_compile_args={
            "cxx": [
                "-O3", "-DNDEBUG"
            ]
        }
    ),
]

if __name__ == "__main__":
    setup(
        name="extensions",
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": CustomBuildExt
        },
        script_args=["build_ext"]
    )
