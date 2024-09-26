import os

if os.getenv("USE_GPU", "false").lower() == "true":
    from . import gpu
    print(">>>> Using GPU backend")
else:
    from . import cpu
    print(">>>> Using CPU backend")
