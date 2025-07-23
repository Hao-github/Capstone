import math

import bitarray
import mmh3


class BloomFilter:
    def __init__(self, n_items, false_positive_rate):
        self.n = n_items
        self.p = false_positive_rate
        self.m = self._get_size(n_items, false_positive_rate)
        self.k = self._get_hash_count(self.m, n_items)
        self.bit_array = bitarray.bitarray(self.m)
        self.bit_array.setall(0)

    def _get_size(self, n, p):
        return int(-n * math.log(p) / (math.log(2) ** 2))

    def _get_hash_count(self, m, n):
        return int((m / n) * math.log(2))

    def add(self, item):
        for i in range(self.k):
            idx = mmh3.hash(item, i) % self.m
            self.bit_array[idx] = 1

    def __contains__(self, item):
        for i in range(self.k):
            idx = mmh3.hash(item, i) % self.m
            if not self.bit_array[idx]:
                return False
        return True
