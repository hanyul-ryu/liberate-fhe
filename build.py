# build.py
import subprocess

# Call setup.py to build CUDA extensions
subprocess.check_call(["python", "setup.py", "build_ext", "--inplace"])
