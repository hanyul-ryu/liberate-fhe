import os

if os.getenv("USE_GPU", "false").lower() == "true":
    print("#######################")
    print(">>>> Using GPU backend")
    print("#######################")
    from .gpu import fhe
else:
    print("#######################")
    print(">>>> Using CPU backend")
    print("#######################")
    from .cpu import fhe
