import numpy as np


class RnsPartition:
    def __init__(
        self, num_ordinary_primes=17, num_special_primes=2, num_devices=1
    ):
        # RNS primes ordered as
        # for the case where
        # num_ordinary_primes = 17
        # num_special_primes = 2
        # num_devices = 3
        # q0, q1, q2, ..., q15, b, p0, p1

        # We would like to partition the ordinary primes as
        # [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16], [17, 18]
        # , where [16] is the base prime, and [17, 18] are the special primes.

        # Then arrange the paritions into 2 devices as
        # lane 0:[[2, 3], [6, 7], [10, 11], [14, 15], [16], [17, 18]]
        # lane 1:[[0, 1], [4, 5], [8, 9], [12, 13], [17, 18]]]

        # For divmod application in key switching and the final rescaling,
        # We 'cache' the last (other than the base prime)
        # alpha partition at every device. This will allow us to avoid
        # 'some' inter-device communications.

        # You will end up using
        # 1. destination_arrays, destination_arrays_with_special: prime indices per a GPU WITHOUT partitioning.
        # 2. parts (p): partitioned indices that start from 0 per a GPU (includes special primes).
        # 3. destination_parts, destination_parts_with_special: parts with indices that are prime indices.
        # 4. part_counts: number of primes in each part.
        # 5. rescaler_loc: the location of the rescaling prime channel, in GPU indices.

        # Populate partitions.
        primes_idx = list(range(num_ordinary_primes - 1))
        base_idx = num_ordinary_primes - 1

        # -(-x//y) = ceil(x/y)
        num_partitions = -(-(num_ordinary_primes - 1) // num_special_primes)

        part = lambda i: primes_idx[
            i * num_special_primes : (i + 1) * num_special_primes
        ]
        partitions = [part(i) for i in range(num_partitions)]

        # Augment the base prime partition.
        partitions.append([num_ordinary_primes - 1])

        # Augment the special prime partition.
        partitions.append(
            list(
                range(
                    num_ordinary_primes,
                    num_ordinary_primes + num_special_primes,
                )
            )
        )

        # Distribute parts.
        # Note that we distribute only the non-base partitions.
        alloc = lambda i: list(
            range(num_partitions - i - 1, -1, -num_devices)
        )[::-1]
        part_allocations = [alloc(i) for i in range(num_devices)]

        # Augment the base prime.
        part_allocations[0].append(num_partitions)

        # Augment the special primes.
        for p in part_allocations:
            p.append(num_partitions + 1)

        # Expand partitions.
        expand_alloc = lambda i: [
            partitions[part] for part in part_allocations[i]
        ]
        prime_allocations = [expand_alloc(i) for i in range(num_devices)]

        # Flatten expanded partitions.
        flat_prime_allocations = [
            sum(alloc, []) for alloc in prime_allocations
        ]

        # Store the results.
        self.num_ordinary_primes = num_ordinary_primes
        self.num_special_primes = num_special_primes
        self.num_devices = num_devices
        self.num_partitions = num_partitions
        self.partitions = partitions
        self.part_allocations = part_allocations
        self.prime_allocations = prime_allocations
        self.flat_prime_allocations = flat_prime_allocations
        self.num_scales = self.num_ordinary_primes - 1

        self.base_prime_idx = self.num_ordinary_primes - 1
        self.special_prime_idx = list(
            range(
                self.num_ordinary_primes + 1,
                self.num_ordinary_primes + 1 + self.num_special_primes,
            )
        )

        # Pre-compute.
        self.compute_destination_arrays()
        self.compute_rescaler_locations()
        self.compute_partitions()

    def compute_destination_arrays(self):
        # Prime channel indices.
        filter_alloc = lambda devi, i: [
            a for a in self.flat_prime_allocations[devi] if a >= i
        ]

        self.destination_arrays_with_special = []
        for lvl in range(self.num_ordinary_primes):
            src = [filter_alloc(devi, lvl) for devi in range(self.num_devices)]
            self.destination_arrays_with_special.append(src)

        special_removed = lambda i: [
            a[: -self.num_special_primes]
            for a in self.destination_arrays_with_special[i]
        ]

        self.destination_arrays = [
            special_removed(i) for i in range(self.num_ordinary_primes)
        ]

        # There may be empty lists.
        # Lint it.
        lint = lambda arr: [a for a in arr if len(a) > 0]
        self.destination_arrays = [lint(a) for a in self.destination_arrays]

    def compute_rescaler_locations(self):
        mins = lambda arr: [min(a) for a in arr]
        mins_loc = lambda a: mins(a).index(min(mins(a)))
        # We use destination_arrays_with_special to prevent empty arrays.
        self.rescaler_loc = [
            mins_loc(a) for a in self.destination_arrays_with_special
        ]

    def partings(self, lvl):
        count_element_sizes = lambda arr: np.array([len(a) for a in arr])
        cumsum_element_sizes = lambda arr: np.cumsum(arr)
        remove_empty_parts = lambda arr: [a for a in arr if a > 0]
        regenerate_parts = lambda arr: [
            list(range(a, b)) for a, b in zip([0] + arr[:-1], arr)
        ]

        part_counts = [count_element_sizes(a) for a in self.prime_allocations]
        part_cumsums = [cumsum_element_sizes(a) for a in part_counts]
        level_diffs = [
            len(a) - len(b)
            for a, b in zip(
                self.destination_arrays_with_special[0],
                self.destination_arrays_with_special[lvl],
            )
        ]

        part_cumsums_lvl = [
            remove_empty_parts(a - d)
            for a, d in zip(part_cumsums, level_diffs)
        ]
        part_count_lvl = [np.diff(a, prepend=0) for a in part_cumsums_lvl]
        parts_lvl = [regenerate_parts(a) for a in part_cumsums_lvl]
        return part_cumsums_lvl, part_count_lvl, parts_lvl

    def compute_partitions(self):
        self.part_cumsums = []
        self.part_counts = []
        self.parts = []
        self.destination_parts = []
        self.destination_parts_with_special = []
        self.p = []
        self.p_special = []
        # Length diff (that is the starting index of the series at level 0)
        self.diff = []

        # self.part_starts = []
        # self.part_ends = []
        # self.part_total_counts = []

        # We frequently use the destination arrays at level 0.
        # Provide quick access to them
        self.d = [
            self.destination_arrays[0][dev_i]
            for dev_i in range(self.num_devices)
        ]

        self.d_special = [
            self.destination_arrays_with_special[0][dev_i]
            for dev_i in range(self.num_devices)
        ]

        for lvl in range(self.num_ordinary_primes):
            pcu, pco, par = self.partings(lvl)
            self.part_cumsums.append(pcu)
            self.part_counts.append(pco)
            self.parts.append(par)

            dest = self.destination_arrays_with_special[lvl]
            destp_special = [
                [[d[pi] for pi in p] for p in dev_p]
                for d, dev_p in zip(dest, par)
            ]
            destp = [dev_dp[:-1] for dev_dp in destp_special]

            self.destination_parts.append(destp)
            self.destination_parts_with_special.append(destp_special)

            # parts start from 0. That is, the indices
            # are directed to the carved out prime series for the level.
            # We may need prime index that are directed to the level 0
            # prime series.
            # We generate such parts, namely p
            diff = [
                len(d1) - len(d2)
                for d1, d2 in zip(
                    self.destination_arrays_with_special[0],
                    self.destination_arrays_with_special[lvl],
                )
            ]
            p_special = [
                [[pi + d for pi in p] for p in dev_p]
                for d, dev_p in zip(diff, self.parts[lvl])
            ]
            p = [dev_p[:-1] for dev_p in p_special]

            self.p.append(p)
            self.p_special.append(p_special)
            self.diff.append(diff)

            # Probably not used
            # part_start = [[0] + a[:-1] for a in pcu]
            # part_end = pcu
            # part_total_count = [len(a) for a in part_start]
            # self.part_starts.append(part_start)
            # self.part_ends.append(part_end)
            # self.part_total_counts.append(part_total_count)
