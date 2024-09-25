from collections import UserDict
from copy import deepcopy

data = {
    "bronze": {
        "logN": 14,
        "num_special_primes": 1,
        # "devices": ["cpu"],
        "scale_bits": 40,
        "num_scales": None,
    },
    "silver": {
        "logN": 15,
        "num_special_primes": 2,
        "devices": ["cpu"],
        "scale_bits": 40,
        "num_scales": None,
    },
    "gold": {
        "logN": 16,
        "num_special_primes": 4,
        # "devices": ["cpu"],
        "scale_bits": 40,
        "num_scales": None,
    },
    "platinum": {
        "logN": 17,
        "num_special_primes": 6,
        # "devices": ["cpu"],
        "scale_bits": 40,
        "num_scales": None,
    },
}


class Params(UserDict):
    def __getitem__(self, key):
        return deepcopy(self.data[key])


params = Params(data)
