import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ["CXX"] = "clang++"

ext_name_csprng = [
    "randround_cpu", "discrete_gaussian_cpu",
    "randint_cpu", "chacha20_cpu"
]


class CustomBuildExt(BuildExtension):
    def build_extension(self, ext):
        path = "src/liberate_cpu/utils/threadpool"
        if ext.name == "ntt_cpu":
            path = "src/liberate_cpu/ntt_cpu/"
        elif ext.name in ext_name_csprng:
            path = "src/liberate_cpu/csprng/"
        elif ext.name == "jthreadpool":
            path = "src/liberate_cpu/utils/threadpool"

        self.build_lib = path

        os.environ["MAX_JOBS"] = str(os.cpu_count())
        super().build_extension(ext)


ext_modules = [
    CppExtension(
        name="jthreadpool",
        sources=["src/liberate_cpu/utils/threadpool/jthreadpool.cpp"],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    ),
    ###################
    ####    ntt    ####
    ###################
    CppExtension(
        name="ntt_cpu",
        sources=[
            "src/liberate_cpu/ntt_cpu/ntt_cpu.cpp"
        ],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    ),
    ######################
    ####    csprng    ####
    ######################
    CppExtension(
        name=ext_name_csprng[0],  # randround
        sources=["src/liberate_cpu/csprng/randround_cpu.cpp"],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    ),
    CppExtension(
        name=ext_name_csprng[1],  # discrete_gaussian
        sources=["src/liberate_cpu/csprng/discrete_gaussian_cpu.cpp"],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    ),
    CppExtension(
        name=ext_name_csprng[2],  # randint
        sources=["src/liberate_cpu/csprng/randint_cpu.cpp"],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    ),
    CppExtension(
        name=ext_name_csprng[3],  # chacha20
        sources=["src/liberate_cpu/csprng/chacha20_cpu.cpp"],
        extra_compile_args=[
            "-std=c++17", "-lstdc++",
            "-Wno-everything", "-I../utils/threadpool/",
            "-O2"
        ]
    )

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
