import math


class Statistics:

    def __init__(self, num, min, max, sum, ssum):
        self.num = num
        self.min = min
        self.max = max
        self.sum = sum
        self.ssum = ssum

    def __add__(self, other):
        self.num += other.num
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.sum += other.sum
        self.ssum += other.ssum
        return self

    def mean(self):
        return self.sum / self.num if self.num != 0 else 0.0

    def variance(self, unbiased=True):
        if self.num <= 1:
            return 0.0

        count = self.num - 1 if unbiased else self.num
        return self.ssum / count - self.sum * self.sum / (self.num * count)

    def std(self, unbiased=True):
        return math.sqrt(max(self.variance(unbiased), 0))
