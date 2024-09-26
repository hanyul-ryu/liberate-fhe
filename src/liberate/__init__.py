import os

if os.getenv("USE_GPU", "false").lower() == "true":
    print("#######################")
    print(">>>> Using GPU backend")
    print("#######################")
    from . import gpu
else:
    print("#######################")
    print(">>>> Using CPU backend")
    print("#######################")
    from . import cpu
