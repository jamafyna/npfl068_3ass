import numpy as np
from collections import Counter, defaultdict


class LinearSmoothedDistribution:
    def __init__(self, data, p_0, p_1, p_2, p_3):
        self.p_0 = p_0
        self.p_1 = defaultdict(lambda: 0, p_1)
        self.p_2 = defaultdict(lambda: 0, p_2)
        self.p_3 = defaultdict(lambda: 0, p_3)
        data = [t for w, t in data]
        trigrams = zip(data, data[1:], data[2:])
        self.lambdas = []
        self.lambdas = self.em_smoothing(trigrams)

    def p(self, t1, t2, t3):
        """Returns the probability of a trigram under the current model"""
        return self.lambdas[3] * self.p_3[t1, t2, t3] + self.lambdas[2] * self.p_2[t2, t3] + \
               self.lambdas[1] * self.p_1[t3] + self.lambdas[0] * self.p_0

    def p_helper(self, t1, t2, t3, lambdas):
        """Returns the probability of a trigram under the current model"""
        return lambdas[3] * self.p_3[t1, t2, t3] + lambdas[2] * self.p_2[t2, t3] + \
               lambdas[1] * self.p_1[t3] + lambdas[0] * self.p_0

    @staticmethod
    def normalize(vector):
        """Returns a normalized vector"""
        s = sum(vector)
        return vector / s

    def em_smoothing(self, trigrams, epsilon=1e-04):
        """Returns the coefficient for smoothed distribution"""
        # initialize lambdas
        lambdas = np.array([0.25, 0.25, 0.25, 0.25])
        # use the held-out data for estimating lambdas
        c_l = np.zeros(4)
        # compute expected counts
        while True:
            for t1, t2, t3 in trigrams:
                p_smoothed = self.p_helper(t1, t2, t3, lambdas)
                c_l[0] += lambdas[0] * self.p_0 / p_smoothed
                c_l[1] += lambdas[1] * self.p_1[t3] / p_smoothed
                c_l[2] += lambdas[2] * self.p_2[t2, t3] / p_smoothed
                c_l[3] += lambdas[3] * self.p_3[t1, t2, t3] / p_smoothed
            # compute next lambdas
            next_l = self.normalize(c_l)
            if all(x < epsilon for x in lambdas - next_l):
                print('DEBUG: EM smoothing converged')
                break
            else:
                lambdas = next_l
            print('DEBUG:', lambdas)
        return lambdas


class MyDefaultDictionary(dict):
    def __init__(self, vocabulary_size):
        self.v = vocabulary_size

    def __missing__(self, key):
        # if there is no such word tag pair, determine from the tag count
        return 1 / (self.get(key[1], 0) + self.v)


class AddOneSmoothedDistribution:
    def __init__(self, training_data, held_out_data):
        # for add 1 smoothing we don't need separate held out data, so we can use everything
        # tags are hidden, therefore we do not need to worry about the tag not being "seen"
        whole_data = training_data + held_out_data
        # word and tag counts
        counts = Counter(whole_data)
        # tag counts
        tag_counts = Counter([t for w, t in whole_data])

        # get the size of the vocabulary
        v = len(counts.keys())
        self.distribution = MyDefaultDictionary(v)
        for w, t in whole_data:
            self.distribution[(w, t)] = (counts[(w, t)] + 1) / (tag_counts[t] + v)

    def p(self, word, tag):
        return self.distribution[(word, tag)]
