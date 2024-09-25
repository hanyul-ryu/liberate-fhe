import datetime
import math
import os
import pickle
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from liberate_cpu.utils.threadpool import jthreadpool

from ..csprng.csprng_cpu import Csprng
from ..ntt_cpu.ntt_context_cpu import NttContext
from .context.ckks_context import CkksContext
from .data_struct import DataStruct
from .encdec import conjugate, decode, encode, rotate
from .presets import errors, types
from .version import VERSION


class ClientCkksEngine:
    @errors.log_error
    def __init__(self,
                 devices: list[int] = None,
                 verbose: bool = False,
                 bias_guard: bool = True,
                 norm: str = 'forward',
                 num_threads=None,
                 thread_load=None,
                 **ctx_params):
        """
            buffer_bit_length=62,
            scale_bits=40,
            logN=15,
            num_scales=None,
            num_special_primes=2,
            sigma=3.2,
            uniform_tenary_secret=True,
            cache_folder='cache/',
            security_bits=128,
            quantum='post_quantum',
            distribution='uniform',
            read_cache=True,
            save_cache=True,
            verbose=False
        """

        self.bias_guard = bias_guard

        self.norm = norm

        self.version = VERSION

        self.__num_threads = num_threads
        if self.__num_threads is None:
            self.__num_threads = 2 ** int(np.floor(np.log2(os.cpu_count() // 2)))

        self.__thread_load = thread_load
        if self.__thread_load is None:
            self.__thread_load = 1

        # Prepare a threadpool.
        self.__tp = jthreadpool.create(self.num_threads)

        self.__ctx = CkksContext(**ctx_params)
        self.__ntt = NttContext(self.ctx, devices=devices, verbose=verbose, num_threads=self.num_threads,
                                thread_load=self.thread_load, tp=self.tp)

        self.__num_levels = self.ntt.num_levels - 1

        self.__num_slots = self.ctx.N // 2

        rng_repeats = max(self.ntt.num_special_primes, 2)
        self.__rng = Csprng(self.ntt.ctx.N, [len(di) for di in self.ntt.p.d], rng_repeats, thread_load=1,
                            num_threads=self.num_threads, devices=self.ntt.devices, tp=self.tp)

        self.__int_scale = 2 ** self.ctx.scale_bits
        self.__scale = np.float64(self.int_scale)

        qstr = ','.join([str(qi) for qi in self.ctx.q])
        hashstr = (self.ctx.generation_string + "_" + qstr).encode("utf-8")
        self.hash = sha256(bytes(hashstr)).hexdigest()

        self.make_adjustments_and_corrections()

        self.device0 = self.ntt.devices[0]

        self.make_mont_PR()

        self.reserve_ksk_buffers()

        # self.create_ksk_rescales()
        #
        # self.alloc_parts()
        #
        # self.leveled_devices()
        #
        # self.create_rescale_scales()
        #
        # self.initialize_key_switching_plan()

        self.__galois_deltas = [2 ** i for i in range(self.ctx.logN - 1)]

    # -------------------------------------------------------------------------------------------
    #   private properties.
    # -------------------------------------------------------------------------------------------
    @property
    def ctx(self):
        return self.__ctx

    @property
    def ntt(self):
        return self.__ntt

    @property
    def rng(self):
        return self.__rng

    @property
    def num_levels(self) -> int:
        return self.__num_levels

    @property
    def num_slots(self) -> int:
        return self.__num_slots

    @property
    def scale(self) -> np.float64:
        return self.__scale

    @property
    def int_scale(self) -> int:
        return self.__int_scale

    @property
    def galois_deltas(self):
        return self.__galois_deltas

    @property
    def num_threads(self):
        return self.__num_threads

    @property
    def thread_load(self):
        return self.__thread_load

    @property
    def tp(self):
        return self.__tp

    def strcp(self, origin):
        return "".join(list(origin).copy())

    # -------------------------------------------------------------------------------------------
    # Various pre-calculations.
    # -------------------------------------------------------------------------------------------

    def create_rescale_scales(self):
        self.rescale_scales = []
        for level in range(self.num_levels):
            self.rescale_scales.append([])

            for device_id in range(self.ntt.num_devices):
                dest_level = self.ntt.p.destination_arrays[level]

                if device_id < len(dest_level):
                    dest = dest_level[device_id]
                    rescaler_device_id = self.ntt.p.rescaler_loc[level]
                    m0 = self.ctx.q[level]

                    if rescaler_device_id == device_id:
                        m = [self.ctx.q[i] for i in dest[1:]]
                    else:
                        m = [self.ctx.q[i] for i in dest]

                    scales = [(pow(m0, -1, mi) * self.ctx.R) % mi for mi in m]

                    scales = torch.tensor(scales,
                                          dtype=self.ctx.torch_dtype,
                                          device=self.ntt.devices[device_id])
                    self.rescale_scales[level].append(scales)

    def leveled_devices(self):
        self.len_devices = []
        for level in range(self.num_levels):
            self.len_devices.append(len([[a] for a in self.ntt.p.p[level] if len(a) > 0]))

        self.neighbor_devices = []
        for level in range(self.num_levels):
            self.neighbor_devices.append([])
            len_devices_at = self.len_devices[level]
            available_devices_ids = range(len_devices_at)
            for src_device_id in available_devices_ids:
                neighbor_devices_at = [
                    device_id for device_id in available_devices_ids if device_id != src_device_id
                ]
                self.neighbor_devices[level].append(neighbor_devices_at)

    def alloc_parts(self):
        self.parts_alloc = []
        for level in range(self.num_levels):
            num_parts = [len(parts) for parts in self.ntt.p.p[level]]
            parts_alloc = [
                alloc[-num_parts[di] - 1:-1] for di, alloc in enumerate(self.ntt.p.part_allocations)
            ]
            self.parts_alloc.append(parts_alloc)

        self.stor_ids = []
        for level in range(self.num_levels):
            self.stor_ids.append([])
            alloc = self.parts_alloc[level]
            min_id = min([min(a) for a in alloc if len(a) > 0])
            for device_id in range(self.ntt.num_devices):
                global_ids = self.parts_alloc[level][device_id]
                new_ids = [i - min_id for i in global_ids]
                self.stor_ids[level].append(new_ids)

    def initialize_key_switching_plan(self):
        self.plan_ids = []
        for level in range(self.num_levels):
            plan_id = self.create_switcher_plan(level)
            self.plan_ids.append(plan_id)

    def create_ksk_rescales(self):
        R = self.ctx.R
        P = self.ctx.q[-self.ntt.num_special_primes:][::-1]
        m = self.ctx.q
        PiR = [[(pow(Pj, -1, mi) * R) % mi for mi in m[:-P_ind - 1]] for P_ind, Pj in enumerate(P)]

        self.PiRs = []

        level = 0
        self.PiRs.append([])

        for P_ind in range(self.ntt.num_special_primes):
            self.PiRs[level].append([])

            for device_id in range(self.ntt.num_devices):
                dest = self.ntt.p.destination_arrays_with_special[level][device_id]
                PiRi = [PiR[P_ind][i] for i in dest[:-P_ind - 1]]
                PiRi = torch.tensor(PiRi,
                                    device=self.ntt.devices[device_id],
                                    dtype=self.ctx.torch_dtype)
                self.PiRs[level][P_ind].append(PiRi)

        for level in range(1, self.num_levels):
            self.PiRs.append([])

            for P_ind in range(self.ntt.num_special_primes):

                self.PiRs[level].append([])

                for device_id in range(self.ntt.num_devices):
                    start = self.ntt.starts[level][device_id]
                    PiRi = self.PiRs[0][P_ind][device_id][start:]

                    self.PiRs[level][P_ind].append(PiRi)

    def reserve_ksk_buffers(self):
        self.ksk_buffers = []

    def make_mont_PR(self):
        P = math.prod(self.ntt.ctx.q[-self.ntt.num_special_primes:])
        R = self.ctx.R
        PR = P * R
        self.mont_PR = []
        for device_id in range(self.ntt.num_devices):
            dest = self.ntt.p.destination_arrays[0][device_id]
            m = [self.ctx.q[i] for i in dest]
            PRm = [PR % mi for mi in m]
            PRm = torch.tensor(PRm,
                               device=self.ntt.devices[device_id],
                               dtype=self.ctx.torch_dtype)
            self.mont_PR.append(PRm)

    def make_adjustments_and_corrections(self):

        self.alpha = [(self.scale / np.float64(q)) ** 2 for q in self.ctx.q[:self.ctx.num_scales]]
        self.deviations = [1]
        for al in self.alpha:
            self.deviations.append(self.deviations[-1] ** 2 * al)

        self.final_q_ind = [da[0][0] for da in self.ntt.p.destination_arrays[:-1]]
        self.final_q = [self.ctx.q[ind] for ind in self.final_q_ind]
        self.final_alpha = [(self.scale / np.float64(q)) for q in self.final_q]
        self.corrections = [1 / (d * fa) for d, fa in zip(self.deviations, self.final_alpha)]

        self.base_prime = self.ctx.q[self.ntt.p.base_prime_idx]

        self.final_scalar = []
        for qi, q in zip(self.final_q_ind, self.final_q):
            scalar = (pow(q, -1, self.base_prime) * self.ctx.R) % self.base_prime
            scalar = torch.tensor([scalar],
                                  device=self.ntt.devices[0],
                                  dtype=self.ctx.torch_dtype)
            self.final_scalar.append(scalar)

        r = self.ctx.R

        self.new_final_scalar = [
            torch.tensor(
                [
                    (pow(self.ctx.q[final_q_ind], -1, base_prime) * r) % base_prime
                    for base_prime in self.ctx.q[final_q_ind + 1: -self.ctx.num_special_primes]
                ],
                device=self.ntt.devices[0],
                dtype=self.ctx.torch_dtype
            ) for final_q_ind in range(self.ntt.num_ordinary_primes - 1)
        ]

    # -------------------------------------------------------------------------------------------
    # Example generation.
    # -------------------------------------------------------------------------------------------

    def absmax_error(self, x, y):
        if isinstance(x[0], np.complex128) and isinstance(y[0], np.complex128):
            # if type(x[0]) == np.complex128 and type(y[0]) == np.complex128:
            r = np.abs(x.real - y.real).max() + np.abs(x.imag - y.imag).max() * 1j
        else:
            r = np.abs(np.array(x) - np.array(y)).max()
        return r

    def integral_bits_available(self):
        base_prime = self.base_prime
        max_bits = math.floor(math.log2(base_prime))
        integral_bits = max_bits - self.ctx.scale_bits
        return integral_bits

    @errors.log_error
    def example(self, amin=None, amax=None, decimal_places: int = 10) -> np.array:
        if amin is None:
            amin = -(2 ** self.integral_bits_available())

        if amax is None:
            amax = 2 ** self.integral_bits_available()

        base = 10 ** decimal_places
        a = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base
        b = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base

        sample = a + b * 1j

        return sample

    # -------------------------------------------------------------------------------------------
    # Encode/Decode
    # -------------------------------------------------------------------------------------------

    def padding(self, m):
        # m = m[:self.num_slots]
        try:
            m_len = len(m)
            padding_result = np.pad(m, (0, self.num_slots - m_len), constant_values=(0, 0))
        except TypeError as e:
            m_len = len([m])
            padding_result = np.pad([m], (0, self.num_slots - m_len), constant_values=(0, 0))
        except Exception as e:
            raise Exception("[Error] encoding Padding Error.")
        return padding_result

    @errors.log_error
    def encode(self, m, level: int = 0, padding=True) -> list[torch.Tensor]:
        """
            Encode a plain message m, using an encoding function.
            Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        deviation = self.deviations[level]
        if padding:
            m = self.padding(m)
        encoded = [encode(m, scale=self.scale, rng=self.rng,
                          device=self.device0,
                          deviation=deviation, norm=self.norm)]

        # pt_buffer = self.ksk_buffers[0][0][0]
        # pt_buffer.copy_(encoded[-1])
        # for dev_id in range(1, self.ntt.num_devices):
        #     encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))
        return encoded

    @errors.log_error
    def decode(self, m, level=0, is_real: bool = False) -> np.ndarray:
        """
            Base prime is located at -1 of the RNS channels in GPU0.
            Assuming this is an orginary RNS deinclude_special.
        """
        correction = self.corrections[level]
        decoded = decode(m[0].squeeze(), scale=self.scale, correction=correction, norm=self.norm)
        m = decoded[:self.ctx.N // 2].cpu().numpy()
        if is_real:
            m = m.real
        return m

    # -------------------------------------------------------------------------------------------
    # secret key/public key generation.
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def create_secret_key(self, include_special: bool = True) -> DataStruct:
        uniform_ternary = self.rng.randint(amax=3, shift=-1, repeats=1)

        mult_type = -2 if include_special else -1
        unsigned_ternary = self.ntt.tile_unsigned(uniform_ternary, lvl=0, mult_type=mult_type)
        self.ntt.enter_ntt(unsigned_ternary, 0, mult_type)

        return DataStruct(
            data=unsigned_ternary,
            include_special=include_special,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["sk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

    @errors.log_error
    def create_public_key(self, sk: DataStruct, include_special: bool = False,
                          crs: list[torch.Tensor] = None) -> DataStruct:
        """
            Generates a public key against the secret key sk.
            pk = -a * sk + e = e - a * sk
        """
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()

        # Set the mult_type
        mult_type = -2 if include_special else -1

        # Generate errors for the ordinary case.
        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        repeats = self.ctx.num_special_primes if sk.include_special else 0

        # Applying mont_mult in the order of 'a', sk will
        if crs is None:
            crs = self.rng.randint(
                self.ntt.q_prepack[mult_type][level][0],
                repeats=repeats
            )

        sa = self.ntt.mont_mult(crs, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)

        return DataStruct(
            data=(pk0, crs),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

    def create_evk(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk2_data = self.ntt.mont_mult(sk.data, sk.data, 0, -2)
        sk2 = DataStruct(
            data=sk2_data,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=sk.level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

        return self.create_key_switching_key(sk2, sk)

    def create_rotation_key(self, sk: DataStruct, delta: int, a: list[torch.Tensor] = None) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

        rotk = self.create_key_switching_key(sk_rotated, sk, crs=a)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def create_galois_key(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        galois_key_parts = [self.create_rotation_key(sk, delta) for delta in self.galois_deltas]

        galois_key = DataStruct(
            data=galois_key_parts,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["galk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )
        return galois_key

    def create_conjugation_key(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [conjugate(s) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )
        rotk = self.create_key_switching_key(sk_rotated, sk)
        rotk = rotk._replace(origin=types.origins["conjk"])
        return rotk

    def create_key_switching_key(self, sk_from: DataStruct, sk_to: DataStruct, crs=None) -> DataStruct:
        """
            Creates a key to switch the key for sk_src to sk_dst.
        """

        if sk_from.origin != types.origins["sk"] or sk_from.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_from.ntt_state) or (not sk_from.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_from.origin)
        if (not sk_to.ntt_state) or (not sk_to.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_to.origin)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_from.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]

        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                a = crs[global_part_id] if crs else None
                pk = self.create_public_key(sk_to, include_special=True, crs=a)

                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                update_part = self.ntt.mont_add([pk_data], [shard], _2q=_2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return DataStruct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level=level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

    # -------------------------------------------------------------------------------------------
    # Encrypt/Decrypt
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def encrypt(self, pt: list[torch.Tensor], pk: DataStruct, level: int = 0) -> DataStruct:
        """
            We again, multiply pt by the scale.
            Since pt is already multiplied by the scale,
            the multiplied pt no longer can be stored
            in a single RNS channel.
            That means we are forced to do the multiplication
            in full RNS domain.
            Note that we allow encryption at
            levels other than 0, and that will take care of multiplying
            the deviation factors.
        """
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(pt, level, mult_type)
        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = DataStruct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

        return ct

    def decrypt_triplet(self, ct_mult: DataStruct, sk: DataStruct, final_round=True) -> list[torch.Tensor]:
        if ct_mult.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=ct_mult.origin, to=types.origins["ctt"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if not ct_mult.ntt_state or not ct_mult.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct_mult.origin)
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct_mult.level
        d0 = [ct_mult.data[0][0].clone()]
        d1 = [ct_mult.data[1][0]]
        d2 = [ct_mult.data[2][0]]

        self.ntt.intt_exit_reduce(d0, level)

        sk_data = [sk.data[0][self.ntt.starts[level][0]:]]

        d1_s = self.ntt.mont_mult(d1, sk_data, level)

        s2 = self.ntt.mont_mult(sk_data, sk_data, level)
        d2_s2 = self.ntt.mont_mult(d2, s2, level)

        self.ntt.intt_exit(d1_s, level)
        self.ntt.intt_exit(d2_s2, level)

        pt = self.ntt.mont_add(d0, d1_s, level)
        pt = self.ntt.mont_add(pt, d2_s2, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct_mult.include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    def decrypt_double(self, ct: DataStruct, sk: DataStruct, final_round: bool = True) -> list[torch.Tensor]:
        """

        @param ct:
        @param sk:
        @param final_round:
        @return:
        """
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct.level

        ct0 = ct.data[0][0]
        sk_data = sk.data[0][self.ntt.starts[level][0]:]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)
        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        if level == self.num_levels - 1:
            base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1

            base = pt[0][base_at][None, :]
            scaler = pt[0][0][None, :]

            final_scalar = self.final_scalar[level]
            scaled = self.ntt.mont_sub([base], [scaler], -1)
            self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
            self.ntt.reduce_2q(scaled, -1)
            self.ntt.make_signed(scaled, -1)

        else:  # level > self.num_levels - 1
            base = pt[0][1:]
            scaler = pt[0][0][None, :]

            final_scalar = self.new_final_scalar[level]

            scaled = base - scaler
            self.ntt.make_unsigned([scaled], lvl=level + 1)
            self.ntt.mont_enter_scalar([scaled], [final_scalar], lvl=level + 1)
            self.ntt.reduce_2q([scaled], lvl=level + 1)

            scaled = scaled[-2:]  # Two channels for decryption

            q1 = self.ctx.q[self.ntt.num_ordinary_primes - 2]
            q0 = self.ctx.q[self.ntt.num_ordinary_primes - 1]

            q1_inv_mod_q0_mont = self.new_final_scalar[self.ntt.num_ordinary_primes - 2]

            quotient = scaled[1:] - scaled[0]
            self.ntt.mont_enter_scalar([quotient], [q1_inv_mod_q0_mont], self.num_levels)
            self.ntt.reduce_2q([quotient], self.num_levels)

            M_half_div_q1 = (q0 - 1) // 2
            M_half_mod_q1 = (q1 - 1) // 2
            is_negative = torch.logical_or(
                quotient[0] > M_half_div_q1,
                torch.logical_and(
                    quotient[0] >= M_half_div_q1, scaled[0] > M_half_mod_q1
                )
            )
            is_negative = is_negative * 1

            signed_large_part = quotient[0] - is_negative * q0
            scaled = [scaled[0].type(torch.float64) + float(q1) * signed_large_part.type(torch.float64)]

        if final_round:
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    def decrypt(self, ct: DataStruct, sk: DataStruct, final_round=True) -> list[torch.Tensor]:
        """
            Decrypt the cipher text ct using the secret key sk.
            Note that the final rescaling must precede the actual decryption process.
        """

        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if ct.origin == types.origins["ctt"]:
            pt = self.decrypt_triplet(ct_mult=ct, sk=sk, final_round=final_round)
        elif ct.origin == types.origins["ct"]:
            pt = self.decrypt_double(ct=ct, sk=sk, final_round=final_round)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=f"{types.origins['ct']} or {types.origins['ctt']}")

        return pt

    # -------------------------------------------------------------------------------------------
    # Key switching.
    # -------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------
    # Rotation.
    # -------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------
    # Fused enc/dec.
    # -------------------------------------------------------------------------------------------
    def encodecrypt(self, m, pk: DataStruct, level: int = 0, padding=True) -> DataStruct:
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

        if padding:
            m = self.padding(m=m)

        deviation = self.deviations[level]
        pt = encode(m, scale=self.scale,
                    device=self.device0, norm=self.norm,
                    deviation=deviation, rng=self.rng,
                    return_without_scaling=self.bias_guard)

        if self.bias_guard:
            dc_integral = pt[0].item() // 1
            pt[0] -= dc_integral

            dc_scale = int(dc_integral) * int(self.scale)
            dc_rns = []
            for device_id, dest in enumerate(self.ntt.p.destination_arrays[level]):
                dci = [dc_scale % self.ctx.q[i] for i in dest]
                dci = torch.tensor(dci,
                                   dtype=self.ctx.torch_dtype,
                                   device=self.ntt.devices[device_id])
                dc_rns.append(dci)

            pt *= np.float64(self.scale)
            pt = self.rng.randround(pt)

        encoded = [pt]

        # pt_buffer = self.ksk_buffers[0][0][0]
        # pt_buffer.copy_(encoded[-1])
        # for dev_id in range(1, self.ntt.num_devices):
        #     encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(encoded, level, mult_type)

        if self.bias_guard:
            for device_id, pti in enumerate(pt_tiled):
                pti[:, 0] += dc_rns[device_id]

        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = DataStruct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level=level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

        return ct

    def __decrode(self, ct: DataStruct, sk: DataStruct, is_real: bool = False, final_round: bool = True) -> np.ndarray:
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct.level
        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1
        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        len_left = len(self.ntt.p.destination_arrays[level][0])

        if (len_left >= 3) and self.bias_guard:
            dc0 = base[0][0].item()
            dc1 = scaler[0][0].item()
            dc2 = pt[0][1][0].item()

            base[0][0] = 0
            scaler[0][0] = 0

            q0_ind = self.ntt.p.destination_arrays[level][0][base_at]
            q1_ind = self.ntt.p.destination_arrays[level][0][0]
            q2_ind = self.ntt.p.destination_arrays[level][0][1]

            q0 = self.ctx.q[q0_ind]
            q1 = self.ctx.q[q1_ind]
            q2 = self.ctx.q[q2_ind]

            Q = q0 * q1 * q2
            Q0 = q1 * q2
            Q1 = q0 * q2
            Q2 = q0 * q1

            Qi0 = pow(Q0, -1, q0)
            Qi1 = pow(Q1, -1, q1)
            Qi2 = pow(Q2, -1, q2)

            dc = (dc0 * Qi0 * Q0 + dc1 * Qi1 * Q1 + dc2 * Qi2 * Q2) % Q

            half_Q = Q // 2
            dc = dc if dc <= half_Q else dc - Q

            dc = (dc + (q1 - 1)) // q1

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        # Decoding.
        correction = self.corrections[level]
        decoded = decode(
            scaled[0][-1],
            scale=self.scale,
            correction=correction,
            norm=self.norm,
            return_without_scaling=self.bias_guard
        )
        decoded = decoded[:self.ctx.N // 2].cpu().numpy()
        ##

        decoded = decoded / self.scale * correction

        # Bias guard.
        if (len_left >= 3) and self.bias_guard:
            decoded += dc / self.scale * correction
        if is_real:
            decoded = decoded.real
        return decoded

    def decryptcode(self, ct: DataStruct, sk: DataStruct, is_real=False, final_round=True) -> np.ndarray:
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        if ct.origin == types.origins["ct"]:
            if self.bias_guard:
                m = self.__decrode(ct=ct, sk=sk, is_real=is_real, final_round=final_round)
            elif self.bias_guard is False:
                pt = self.decrypt(ct=ct, sk=sk, final_round=final_round)
                m = self.decode(m=pt, level=ct.level, is_real=is_real)
            else:
                raise "Error"
        elif ct.origin == types.origins["ctt"]:
            pt = self.decrypt_triplet(ct_mult=ct, sk=sk, final_round=final_round)
            m = self.decode(m=pt, level=ct.level, is_real=is_real)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        return m

    # -------------------------------------------------------------------------------------------
    # Shortcuts
    # -------------------------------------------------------------------------------------------
    def encorypt(self, m, pk: DataStruct, level: int = 0, padding=True):
        return self.encodecrypt(m, pk=pk, level=level, padding=padding)

    def decrode(self, ct: DataStruct, sk: DataStruct, is_real=False, final_round=True):
        return self.decryptcode(ct=ct, sk=sk, is_real=is_real, final_round=final_round)

    # -------------------------------------------------------------------------------------------
    # Clone.
    # -------------------------------------------------------------------------------------------

    def clone_tensors(self, data: DataStruct) -> DataStruct:
        new_data = []
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            for device_data in data:
                new_data.append(device_data.clone())
        else:
            for part in data:
                new_data.append([])
                for device_data in part:
                    new_data[-1].append(device_data.clone())
        return new_data

    def clone(self, text):
        if not isinstance(text.data[0], DataStruct):
            # data are tensors.
            data = self.clone_tensors(text.data)

            wrapper = DataStruct(
                data=data,
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                level_available=self.num_levels,
                hash=text.hash,
                version=text.version
            )

        else:
            wrapper = DataStruct(
                data=[],
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                level_available=self.num_levels,
                hash=text.hash,
                version=text.version
            )

            for d in text.data:
                wrapper.data.append(self.clone(d))

        return wrapper

        # -------------------------------------------------------------------------------------------
        # Move data back and forth from GPUs to the CPU.
        # -------------------------------------------------------------------------------------------

    def download_to_cpu(self, gpu_data, level, include_special):
        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        # Tensor size parameters.
        num_rows = sum([len(d) for d in dest])
        num_cols = self.ctx.N
        cpu_size = (num_rows, num_cols)

        # Make a cpu tensor to aggregate the data in GPUs.
        cpu_tensor = torch.empty(cpu_size, dtype=self.ctx.torch_dtype, device='cpu')

        for ten, dest_i in zip(gpu_data, dest):
            # Check if the tensor is in the gpu.
            if ten.device.type != 'cuda':
                raise Exception("To download data to the CPU, it must already be in a GPU!!!")

            # Copy in the data.
            cpu_tensor[dest_i] = ten.cpu()

        # To avoid confusion, make a list with a single element (only one device, that is the CPU),
        # and return it.
        return [cpu_tensor]

    def upload_to_gpu(self, cpu_data, level, include_special):
        # There's only one device data in the cpu data.
        cpu_tensor = cpu_data[0]

        # Check if the tensor is in the cpu.
        if cpu_tensor.device.type != 'cpu':
            raise Exception("To upload data to GPUs, it must already be in the CPU!!!")

        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        gpu_data = []
        for device_id in range(len(dest)):
            # Copy in the data.
            dest_device = dest[device_id]
            device = self.ntt.devices[device_id]
            gpu_tensor = cpu_tensor[dest_device].to(device=device)

            # Append to the gpu_data list.
            gpu_data.append(gpu_tensor)

        return gpu_data

    def move_tensors(self, data, level, include_special, direction):
        func = {
            'gpu2cpu': self.download_to_cpu,
            'cpu2gpu': self.upload_to_gpu
        }[direction]

        # Some data has 1 depth.
        if not isinstance(data[0], list):
            moved = func(data, level, include_special)
            new_data = moved
        else:
            new_data = []
            for part in data:
                moved = func(part, level, include_special)
                new_data.append(moved)
        return new_data

    def move_to(self, text, direction='gpu2cpu'):
        if not isinstance(text.data[0], DataStruct):
            level = text.level
            include_special = text.include_special

            # data are tensors.
            data = self.move_tensors(text.data, level,
                                     include_special, direction)

            wrapper = DataStruct(
                data=data,
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                level_available=self.num_levels,
                hash=text.hash,
                version=text.version
            )

        else:
            wrapper = DataStruct(
                data=[],
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level=text.level,
                level_available=self.num_levels,
                hash=text.hash,
                version=text.version
            )

            for d in text.data:
                moved = self.move_to(d, direction)
                wrapper.data.append(moved)

        return wrapper

        # Shortcuts

    def cpu(self, ct):
        return self.move_to(ct, 'gpu2cpu')

    def cuda(self, ct):
        return self.move_to(ct, 'cpu2gpu')

        # -------------------------------------------------------------------------------------------
        # check device.
        # -------------------------------------------------------------------------------------------

    def tensor_device(self, data):
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            return data[0].device.type
        else:
            return data[0][0].device.type

    def device(self, text):
        if not isinstance(text.data[0], DataStruct):
            # data are tensors.
            return self.tensor_device(text.data)
        else:
            return self.device(text.data[0])

        # -------------------------------------------------------------------------------------------
        # Print data structure.
        # -------------------------------------------------------------------------------------------

    def tree_lead_text(self, level, tabs=2, final=False):
        final_char = "└" if final else "├"

        if level == 0:
            leader = " " * tabs
            trailer = "─" * tabs
            lead_text = "─" * tabs + "┬" + trailer

        elif level < 0:
            level = -level
            leader = " " * tabs
            trailer = "─" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + final_char + trailer

        else:
            leader = " " * tabs
            trailer = "┬" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + "├" + trailer

        return lead_text

    def print_data_shapes(self, data, level):
        # Some data structures have 1 depth.
        if isinstance(data[0], list):
            for part_i, part in enumerate(data):
                for device_id, d in enumerate(part):
                    device = self.ntt.devices[device_id]

                    if (device_id == len(part) - 1) and \
                            (part_i == len(data) - 1):
                        final = True
                    else:
                        final = False

                    lead_text = self.tree_lead_text(-level, final=final)

                    print(f"{lead_text} tensor at device {device} with "
                          f"shape {d.shape}.")
        else:
            for device_id, d in enumerate(data):
                device = self.ntt.devices[device_id]

                if device_id == len(data) - 1:
                    final = True
                else:
                    final = False

                lead_text = self.tree_lead_text(-level, final=final)

                print(f"{lead_text} tensor at device {device} with "
                      f"shape {d.shape}.")

    def print_data_structure(self, text, level=0):
        lead_text = self.tree_lead_text(level)
        print(f"{lead_text} {text.origin}")

        if not isinstance(text.data[0], DataStruct):
            self.print_data_shapes(text.data, level + 1)
        else:
            for d in text.data:
                self.print_data_structure(d, level + 1)

    # -------------------------------------------------------------------------------------------
    # Save and load.
    # -------------------------------------------------------------------------------------------

    def auto_generate_filename(self, fmt_str='%Y%m%d%H%M%s%f'):
        return datetime.datetime.now().strftime(fmt_str) + '.pkl'

    def save(self, text, filename=None):
        if filename is None:
            filename = self.auto_generate_filename()

        savepath = Path(filename)

        # Check if the text is in the CPU.
        # If not, move to CPU.
        if self.device(text) != 'cpu':
            cpu_text = self.cpu(text)
        else:
            cpu_text = text

        with savepath.open('wb') as f:
            pickle.dump(cpu_text, f)

    def load(self, filename):
        savepath = Path(filename)
        with savepath.open('rb') as f:
            cpu_text = pickle.load(f)

        text = cpu_text

        return text

    # -------------------------------------------------------------------------------------------
    # Multiparty.
    # -------------------------------------------------------------------------------------------
    def multiparty_public_crs(self, pk: DataStruct) -> list[torch.tensor]:
        """
            Get public key's CRS.
        @param pk:
        @return:
        """
        crs = self.clone(pk).data[1]
        return crs

    def multiparty_create_public_key(self,
                                     sk: DataStruct,
                                     crs: list[torch.tensor],
                                     include_special: bool = False) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()
        mult_type = -2 if include_special else -1

        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        # repeats = self.ctx.num_special_primes if sk.include_special else 0
        # if crs is None:
        #     crs = self.rng.randint(
        #         self.ntt.q_prepack[mult_type][level][0],
        #         repeats=repeats
        #     )

        sa = self.ntt.mont_mult(crs, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)
        pk = DataStruct(
            data=(pk0, crs),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level=level,
            hash=self.hash,
            level_available=self.num_levels,
            version=self.version
        )
        return pk

    def multiparty_create_collective_public_key(self, pks: list[DataStruct]) -> DataStruct:
        data, include_special, ntt_state, montgomery_state, origin, level, hash_, level_available, version = pks[0]
        mult_type = -2 if include_special else -1
        b = [b.clone() for b in data[0]]  # num of gpus
        a = [a.clone() for a in data[1]]

        for pk in pks[1:]:
            b = self.ntt.mont_add(b, pk.data[0], lvl=0, mult_type=mult_type)

        cpk = DataStruct(
            (b, a),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["pk"],
            level=level,
            level_available=level_available,
            hash=self.hash,
            version=self.version
        )
        return cpk

    def multiparty_decrypt_head(self, ct: DataStruct, sk: DataStruct):
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)
        level = ct.level

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)

        return pt

    def multiparty_decrypt_partial(self, ct: DataStruct, sk: DataStruct) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        data, include_special, ntt_state, montgomery_state, origin, level, hash_, _, version = ct

        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        return sa

    def multiparty_decrypt_fusion(self, pcts: list, level=0, include_special=False):
        pt = [x.clone() for x in pcts[0]]
        for pct in pcts[1:]:
            pt = self.ntt.mont_add(pt, pct, level)

        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        m = self.decode(m=scaled, level=level)

        return m

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. ROTATION
    #### -------------------------------------------------------------------------------------------

    def multiparty_create_key_switching_key(self, sk_src: DataStruct, sk_dst: DataStruct, crs=None) -> DataStruct:
        if sk_src.origin != types.origins["sk"] or sk_src.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_src.ntt_state) or (not sk_src.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_src.origin)
        if (not sk_dst.ntt_state) or (not sk_dst.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_dst.origin)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_src.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]
        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                a = crs[global_part_id] if crs else None
                pk = self.multiparty_create_public_key(sk_dst, include_special=True, crs=a)
                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                # update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                update_part = self.ntt.mont_add([pk_data], [shard], _2q=_2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                # Name the pk.
                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return DataStruct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level=level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

    def multiparty_create_rotation_key(self, sk: DataStruct, delta: int, crs=None) -> DataStruct:
        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )
        rotk = self.multiparty_create_key_switching_key(sk_rotated, sk, crs=crs)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def multiparty_generate_rotation_key(self, rotks: list[DataStruct]) -> DataStruct:
        crotk = self.clone(rotks[0])
        for rotk in rotks[1:]:
            for ksk_idx in range(len(rotk.data)):
                update_parts = self.ntt.mont_add(crotk.data[ksk_idx].data[0], rotk.data[ksk_idx].data[0])
                crotk.data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return crotk

    def generate_rotation_crs(self, rotk: DataStruct):
        if types.origins["rotk"] not in rotk.origin and types.origins["ksk"] != rotk.origin:
            raise errors.NotMatchType(origin=rotk.origin, to=types.origins["ksk"])
        crss = []
        for ksk in rotk.data:
            crss.append(ksk.data[1])
        return crss

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. GALOIS
    #### -------------------------------------------------------------------------------------------

    def generate_galois_crs(self, galk: DataStruct):
        if galk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=galk.origin, to=types.origins["galk"])
        crs_s = []
        for rotk in galk.data:
            crss = [ksk.data[1] for ksk in rotk.data]
            crs_s.append(crss)
        return crs_s

    def multiparty_create_galois_key(self, sk: DataStruct, a: list) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        galois_key_parts = [
            self.multiparty_create_rotation_key(sk, self.galois_deltas[idx], crs=a[idx])
            for idx in range(len(self.galois_deltas))
        ]

        galois_key = DataStruct(
            data=galois_key_parts,
            include_special=True,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["galk"],
            level=0,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )
        return galois_key

    def multiparty_generate_galois_key(self, galks: list[DataStruct]) -> DataStruct:
        cgalk = self.clone(galks[0])
        for galk in galks[1:]:  # galk
            for rotk_idx in range(len(galk.data)):  # rotk
                for ksk_idx in range(len(galk.data[rotk_idx].data)):  # ksk
                    update_parts = self.ntt.mont_add(
                        cgalk.data[rotk_idx].data[ksk_idx].data[0],
                        galk.data[rotk_idx].data[ksk_idx].data[0]
                    )
                    cgalk.data[rotk_idx].data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return cgalk

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. Evaluation Key
    #### -------------------------------------------------------------------------------------------

    def multiparty_sum_evk_share(self, evks_share: list[DataStruct]):
        evk_sum = self.clone(evks_share[0])
        for evk_share in evks_share[1:]:
            for ksk_idx in range(len(evk_sum.data)):
                update_parts = self.ntt.mont_add(evk_sum.data[ksk_idx].data[0], evk_share.data[ksk_idx].data[0])
                for dev_id in range(len(update_parts)):
                    evk_sum.data[ksk_idx].data[0][dev_id].copy_(update_parts[dev_id], non_blocking=True)

        return evk_sum

    def multiparty_mult_evk_share_sum(self, evk_sum: DataStruct, sk: DataStruct):
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        evk_sum_mult = self.clone(evk_sum)

        for ksk_idx in range(len(evk_sum.data)):
            update_part_b = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[0], sk.data)
            update_part_a = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[1], sk.data)
            for dev_id in range(len(update_part_b)):
                evk_sum_mult.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                evk_sum_mult.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)

        return evk_sum_mult

    def multiparty_sum_evk_share_mult(self, evk_sum_mult: list[DataStruct]) -> DataStruct:
        cevk = self.clone(evk_sum_mult[0])
        for evk in evk_sum_mult[1:]:
            for ksk_idx in range(len(cevk.data)):
                update_part_b = self.ntt.mont_add(cevk.data[ksk_idx].data[0], evk.data[ksk_idx].data[0])
                update_part_a = self.ntt.mont_add(cevk.data[ksk_idx].data[1], evk.data[ksk_idx].data[1])
                for dev_id in range(len(update_part_b)):
                    cevk.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                    cevk.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)
        return cevk
