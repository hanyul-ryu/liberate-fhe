import subprocess

import torch

if torch.cuda.is_available():
    # Call setup_gpu.py to build CUDA extensions
    print("#####################################################")
    print(">>>> CUDA is available. Build GPU extensions.")
    print("#####################################################")
    subprocess.check_call(["python", "setup_gpu.py", "build_ext", "--inplace"])

else:
    # Call setup_cpu.py to build C++ extensions
    print("#####################################################")
    print(">>>> CUDA is not available. Build CPU extensions.")
    print("#####################################################")
    subprocess.check_call(["python", "setup_cpu.py", "build_ext", "--inplace"])
