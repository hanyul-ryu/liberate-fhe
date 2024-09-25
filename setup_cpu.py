import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ["CXX"] = "clang++"

ext_name_csprng = [
    "randround_cpu",
    "discrete_gaussian_cpu",
    "randint_cpu",
    "chacha20_cpu",
]
path_source = "src/liberate/cpu"


class CustomBuildExt(BuildExtension):
    def build_extension(self, ext):
        path = os.path.join(path_source, "utils/threadpool")
        if ext.name == "ntt_cpu":
            path = os.path.join(path_source, "ntt_cpu")
        elif ext.name in ext_name_csprng:
            path = os.path.join(path_source, "csprng")
        elif ext.name == "jthreadpool":
            path = os.path.join(path_source, "utils/threadpool")

        self.build_lib = path

        os.environ["MAX_JOBS"] = str(os.cpu_count())
        super().build_extension(ext)


ext_modules = [
    CppExtension(
        name="jthreadpool",
        sources=[
            os.path.join(path_source, "utils/threadpool/jthreadpool.cpp")
        ],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
    ###################
    ####    ntt    ####
    ###################
    CppExtension(
        name="ntt_cpu",
        sources=[os.path.join(path_source, "ntt_cpu/ntt_cpu.cpp")],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
    ######################
    ####    csprng    ####
    ######################
    CppExtension(
        name=ext_name_csprng[0],  # randround
        sources=[os.path.join(path_source, "csprng/randround_cpu.cpp")],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
    CppExtension(
        name=ext_name_csprng[1],  # discrete_gaussian
        sources=[
            os.path.join(path_source, "csprng/discrete_gaussian_cpu.cpp")
        ],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
    CppExtension(
        name=ext_name_csprng[2],  # randint
        sources=[os.path.join(path_source, "csprng/randint_cpu.cpp")],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
    CppExtension(
        name=ext_name_csprng[3],  # chacha20
        sources=[os.path.join(path_source, "csprng/chacha20_cpu.cpp")],
        extra_compile_args=[
            "-std=c++17",
            "-lstdc++",
            "-Wno-everything",
            "-I../utils/threadpool/",
            "-O2",
        ],
    ),
]

if __name__ == "__main__":
    setup(
        name="extensions",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        script_args=["build_ext"],
    )
