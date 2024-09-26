import subprocess
import torch

if torch.cuda.is_available():
    # Call setup_gpu.py to build CUDA extensions
    subprocess.check_call(["python", "setup_gpu.py", "build_ext", "--inplace"])
else:
    # Call setup_cpu.py to build C++ extensions
    subprocess.check_call(["python", "setup_cpu.py", "build_ext", "--inplace"])
