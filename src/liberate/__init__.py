from torch import cuda

if cuda.is_available():
    print("#######################")
    print(">>>> Using GPU backend")
    print("#######################")
    from .gpu import fhe
else:
    print("#######################")
    print(">>>> Using CPU backend")
    print("#######################")
    from .cpu import fhe
