import os
import shutil
import subprocess

import torch

if torch.cuda.is_available():
    # Call build_gpu.py to build CUDA extensions
    print("#######################################################")
    print("#### >>>> CUDA is available. Build GPU extensions. ####")
    print("#######################################################")
    subprocess.check_call(["python", "build_gpu.py", "build_ext", "--inplace"])

    path_folder = "src/liberate/cpu"
    if os.path.exists(path_folder):
        shutil.rmtree(path_folder)

else:
    # Call build_cpu.py to build C++ extensions
    print("###########################################################")
    print("#### >>>> CUDA is not available. Build CPU extensions. ####")
    print("###########################################################")
    subprocess.check_call(["python", "build_cpu.py", "build_ext", "--inplace"])

    path_folder = "src/liberate/gpu"
    if os.path.exists(path_folder):
        shutil.rmtree(path_folder)
