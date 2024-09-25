import binascii
import os

import numpy as np
import torch

from liberate.cpu.utils.threadpool import jthreadpool

from .chacha20_cpu import chacha20_cpu
from .discrete_gaussian_cpu import discrete_gaussian_cpu
from .discrete_gaussian_sampler import build_CDT_binary_search_tree
from .randint_cpu import randint_cpu
from .randround_cpu import randround_cpu

torch.backends.cudnn.benchmark = True


def is_power_of_two(n):
    if n == 0:
        return False
    while n != 1:
        if n % 2 != 0:
            return False
        n = n // 2
    return True


class Csprng:
    def __init__(
        self,
        num_coefs=2**15,
        num_channels=[8],
        num_repeating_channels=2,
        sigma=3.2,
        seed=None,
        nonce=None,
        thread_load=1,
        num_threads=None,
        devices=None,
        tp=None,
        **kwargs,
    ):
        """N is the length of the polynomial, and C is the number of RNS channels.
        procure the maximum (at level zero, special multiplication) at initialization.
        """

        # This CSPRNG class generates
        # 1. num_coefs x (num_channels + num_repeating_channels) uniform distributed
        #    random numbers at max. num_channels can be reduced down according
        #    to the input q at the time of generation.
        #    The numbers generated ri are 0 <= ri < qi.
        #    Seeds in the repeated channels are the same, and hence in those
        #    channels, the generated numbers are the same across GPUs.
        #    Generation of the repeated random numbers is optional.
        #    The same function can be used to generate ranged random integers
        #    in a fixed range. Again, generation of the repeated numbers is optional.
        # 2. Generation of Discrete Gaussian random numbers. The numbers can be generated
        #    in the non-repeating channels (with the maximum number of channels num_channels),
        #    or in the repeating channels (with the maximum number of
        #    channels num_repeating_channels, where in the most typical scenario is same as 1).

        self.num_coefs = num_coefs
        self.num_channels = num_channels
        self.num_repeating_channels = num_repeating_channels
        self.sigma = sigma

        self.thread_load = thread_load
        if thread_load is None:
            self.thread_load = 1

        # Number of threads to use.
        self.num_threads = num_threads
        if num_threads is None:
            self.num_threads = 2 ** int(np.floor(np.log2(os.cpu_count() // 2)))

        # Make sure num_thread is a power of two.
        if not is_power_of_two(self.num_threads):
            raise Exception("Number of threads must be a power of two.")

        self.shares = [num_channels[0]]

        # We generate random bytes 4x4 = 16 per an array and hence,
        # internally only need to procure N // 4 length arrays.
        # Out of the 16, we generate discrete gaussian or uniform
        # samples 4 at a time.
        self.L = self.num_coefs // 4

        # We build binary search tree for discrete gaussian here.
        (
            self.btree,
            self.btree_ptr,
            self.btree_size,
            self.tree_depth,
        ) = build_CDT_binary_search_tree(security_bits=128, sigma=sigma)

        # Total increment to add to counters after each random bytes generation.
        self.inc = (
            self.num_channels[0] + self.num_repeating_channels
        ) * self.L

        # expand 32-byte k.
        # This is 1634760805, 857760878, 2036477234, 1797285236.
        str2ord = lambda s: sum([2 ** (i * 8) * c for i, c in enumerate(s)])
        str_constant = torch.tensor(
            [
                str2ord(b"expa"),
                str2ord(b"nd 3"),
                str2ord(b"2-by"),
                str2ord(b"te k"),
            ],
            dtype=torch.int64,
        )
        self.nothing_up_my_sleeve = str_constant

        # Prepare a state tensor.
        state_size = (
            (self.num_channels[0] + self.num_repeating_channels) * self.L,
            16,
        )
        self.state = torch.zeros(state_size, dtype=torch.int64)

        # Prepare a channeled views.
        self.channeled_state = self.state.view(
            (self.num_channels[0] + self.num_repeating_channels), self.L, -1
        )

        # The counter.
        counter_list = list(range(0, self.inc))
        self.counter = torch.tensor(counter_list, dtype=torch.int64)

        self.refresh(seed, nonce)

        # Prepare a threadpool.
        if tp is None:
            self.tp = jthreadpool.create(self.num_threads)
            self.global_tp = False
        else:
            self.tp = tp
            self.global_tp = True

    # Tidy up.
    def __del__(self):
        if not self.global_tp:
            jthreadpool.clean(self.tp)

    def refresh(self, seed=None, nonce=None):
        # Generate seed if necessary.
        self.key = self.generate_key(seed)

        # Generate nonce if necessary.
        self.nonce = self.generate_nonce(nonce)

        # Iterate over all devices.
        self.initialize_state(seed, nonce)

    def initialize_state(self, seed=None, nonce=None):
        state = self.state
        state.zero_()

        # Set the counter.
        # It is hardly unlikely we will use CxL > 2**32.
        # Just fill in the 12th element
        # (The lower bytes of the counter).
        state[:, 12] = self.counter[None, :]

        # Set the expand 32-byte k
        state[:, 0:4] = self.nothing_up_my_sleeve[None, :]

        # Set the seed.
        state[:, 4:12] = self.key[None, :]

        # Fill in nonce.
        state[:, 14:] = self.nonce[None, :]

    def generate_initial_bytes(self, nbytes, part_bytes=4, seed=None):
        if seed is None:
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes: int(binascii.hexlify(x), 16)
            seed0 = [
                hex2int(os.urandom(part_bytes), part_bytes)
                for _ in range(n_keys)
            ]
            seed_tensor = torch.tensor(seed0, dtype=torch.int64)
        else:
            seed0 = seed
            seed_tensor = torch.tensor(seed0, dtype=torch.int64)

        return seed_tensor

    def generate_key(self, seed):
        # 256bits seed as a key.
        # We generate the same key seed for every GPU.
        # Randomity is produced by counters, not the key.
        return self.generate_initial_bytes(32, seed=None)

    def generate_nonce(self, seed):
        # nonce is 64bits.
        return self.generate_initial_bytes(8, seed=None)

    def randbytes(
        self, num_channels=None, length=None, reshape=False, repeats=1
    ):
        if length is None:
            L = self.L
        else:
            L = length // 16

        if num_channels is None:
            C = self.num_channels
        else:
            C = num_channels

        # Set the target state.
        target_state = (
            self.channeled_state[:C, :L, :].view(-1, 16).contiguous()
        )

        # Derive random bytes.
        chunk = C * L // (self.num_threads * self.thread_load)
        random_bytes = chacha20_cpu(target_state, self.inc, chunk, self.tp)

        # If not reshape, flatten.
        if reshape:
            random_bytes = random_bytes.view(C, L, 16)

        return [random_bytes]

    def randint(self, amax=3, shift=0, length=None, repeats=1):
        if not isinstance(amax, (list, tuple)):
            amax = [[amax] for share in self.shares]

        if length is None:
            L = self.L
        else:
            L = length // 4

        # Figure out the number of channels C.
        C = len(amax[0])

        # Calculate shares.
        # If repeats are greater than 0, those channels are
        # subtracted from shares.
        shares = [len(am) - repeats for am in amax]

        # Convert the amax list to contiguous numpy array pointers.
        q_conti = np.ascontiguousarray(amax[0], dtype=np.uint64)
        q_ptr = q_conti.__array_interface__["data"][0]

        # Set the target states.
        target_states = []
        start_channel = self.shares[0] - shares[0]
        end_channel = self.shares[0] + repeats
        target_state = self.channeled_state[
            start_channel:end_channel, :L, :
        ].contiguous()

        chunk = (
            (end_channel - start_channel)
            * L
            // (self.num_threads * self.thread_load)
        )
        result = randint_cpu(
            target_state, q_ptr, shift, self.inc, chunk, self.tp
        )

        return [result]

    def discrete_gaussian(
        self, non_repeats=0, repeats=1, length=None, reshape=False
    ):
        if not isinstance(non_repeats, (list, tuple)):
            shares = [non_repeats]
        else:
            shares = non_repeats

        if length is None:
            L = self.L
        else:
            L = length // 4

        # Set the target states.
        target_states = []
        start_channel = self.shares[0] - shares[0]
        end_channel = self.shares[0] + repeats
        target_state = self.channeled_state[
            start_channel:end_channel, :L, :
        ].contiguous()
        C = target_state.size(0)
        chunk = C * L // (self.num_threads * self.thread_load)

        dg = discrete_gaussian_cpu(
            target_state,
            self.inc,
            self.btree_ptr,
            self.btree_size,
            self.tree_depth,
            chunk,
            self.tp,
        )
        dg = [dg.view(-1, self.num_coefs)]

        return dg

    def randround(self, coef):
        poly_length = len(coef)
        dcoef = coef.to(torch.double).contiguous()
        M = poly_length // 16
        target_state = self.state[:M, :].contiguous()
        chunk = M // (self.num_threads * self.thread_load)

        # Chunk size less than 128 is trivial and can deteriorate the performance.
        chunk = 128 if chunk < 128 else chunk
        rounded = randround_cpu(target_state, dcoef, self.inc, chunk, self.tp)
        return rounded
