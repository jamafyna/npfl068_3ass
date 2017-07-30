import numpy as np


class LinearSmoothedDistribution:
    def __init__(self, data, p_0, p_1, p_2, p_3):
        self.p_0 = p_0
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3
        trigrams = zip(data, data[1:], data[2:])
        self.lambdas = self.em_smoothing(trigrams, p_0, p_1, p_2, p_3)

    def p(self, t1, t2, t3):
        """Returns the probability of a trigram under the current model"""
        return self.lambdas[3] * self.p_3[t1][t2][t3] + self.lambdas[2] * self.p_2[t2][t3] + self.lambdas[1] * self.p_1[
            t3] + self.lambdas[0] * self.p_0

    @staticmethod
    def normalize(vector):
        """Returns a normalized vector"""
        s = sum(vector)
        return vector / s

    def em_smoothing(self, trigrams, p_0, p_1, p_2, p_3, epsilon=1e-03):
        """Returns the coefficient for smoothed distribution"""
        # initialize lambdas
        lambdas = np.array([0.25, 0.25, 0.25, 0.25])
        # use the held-out data for estimating lambdas
        c_l = np.zeros(4)
        # compute expected counts
        while True:
            for t1, t2, t3 in trigrams:
                p_smoothed = self.p(lambdas, t1, t2, t3, p_0, p_1, p_2, p_3)
                c_l[0] += lambdas[0] * self.p_0 / p_smoothed
                c_l[1] += lambdas[1] * self.p_1[t3] / p_smoothed
                c_l[2] += lambdas[2] * self.p_2[t2][t3] / p_smoothed
                c_l[3] += lambdas[3] * self.p_3[t1][t2][t3] / p_smoothed
            # compute next lambdas
            next_l = self.normalize(c_l)
            if all(x < epsilon for x in lambdas - next_l):
                print('DEBUG: EM smoothing converged')
                break
            else:
                lambdas = next_l
            print('DEBUG:', lambdas)
        return lambdas
