from collections import Counter, defaultdict
from sys import stdin as sin
from sys import stdout as sout


class LinearSmoothedDistribution:
    def __init__(self, held_out_data, p_0, p_1, p_2, p_3):
        self.p_0 = p_0
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3
        held_out_data = [t for w, t in held_out_data]
        trigrams = list(zip(held_out_data, held_out_data[1:], held_out_data[2:]))
        self.lambdas = []
        self.lambdas = self.em_smoothing(trigrams)
        # store precomputed values
        self.distribution = Counter()

    def p(self, t1, t2, t3):
        """Returns the probability of a trigram under the current model"""
        # if the trigram was already precomputed just return it
        if (t1, t2, t3) in self.distribution:
            return self.distribution[t1, t2, t3]
        # else compute, store and return it
        self.distribution[t1, t2, t3] = self.lambdas[3] * self.p_3[t1, t2, t3] + self.lambdas[2] * self.p_2[
            t2, t3] + self.lambdas[1] * self.p_1[t3] + self.lambdas[0] * self.p_0
        return self.distribution[t1, t2, t3]

    def _compute_p(self, t1, t2, t3, lambdas):
        """Returns the probability of a trigram under the current model"""
        return lambdas[3] * self.p_3[t1, t2, t3] + lambdas[2] * self.p_2[t2, t3] + \
               lambdas[1] * self.p_1[t3] + lambdas[0] * self.p_0

    @staticmethod
    def _normalize(vector):
        """Returns a normalized vector"""
        s = sum(vector)
        return [e / s for e in vector]

    def em_smoothing(self, trigrams, epsilon=1e-03):
        """Returns the coefficient for smoothed distribution"""
        # initialize lambdas
        lambdas = [0.25, 0.25, 0.25, 0.25]
        # use the held-out data for estimating lambdas
        # c_l = np.zeros(4)
        # compute expected counts
        while True:
            c_l = [0, 0, 0, 0]
            for t1, t2, t3 in trigrams:
                if t3 == '###':
                    continue
                p_smoothed = self._compute_p(t1, t2, t3, lambdas)
                c_l[0] += lambdas[0] * self.p_0 / p_smoothed
                c_l[1] += lambdas[1] * self.p_1[t3] / p_smoothed
                c_l[2] += lambdas[2] * self.p_2[t2, t3] / p_smoothed
                c_l[3] += lambdas[3] * self.p_3[t1, t2, t3] / p_smoothed
            # print(t1, t2, t3, ':', c_l)
            # compute next lambdas
            next_l = self._normalize(c_l)
            if all(x < epsilon for x in [lambdas[i] - next_l[i] for i in range(4)]):
                print('INFO: EM smoothing converged:', lambdas)
                break
            else:
                lambdas = next_l
        return lambdas


class Pwt:
    """
    Computes the initial distribution of p(w|t) and smooths it with add less than one smoothing
    """

    def __init__(self, data, wordset_size, tagset_size):
        self.t_counts = Counter([t for (_, t) in data])
        # self.vocabulary = set([w for (w, _) in data])
        self.wt_counts = Counter(data)
        self.vocab_size = wordset_size * tagset_size
        self.distribution = defaultdict(lambda: 0)
        # self.tagset_len = tagset_size

    # def get_smoothed_pwt(self, w, t, isOOV, lamb=2 ** (-10)):
    #     """
    #     Returns smoothed p(w|t), by smoothing less than 1. Suppose that w and t are from known wordset and tagset, not unknown.
    #     """
    #     # if the w is out-of-vocabulary, then use uniform distribution
    #     if isOOV:
    #         return 1 / self.tagset_len
    #     return (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)

    def p(self, w, t, lamb=2 ** (-10)):
        """
        Returns smoothed p(w|t) by adding less than 1. Suppose that w and t are from known wordset and tagset, not unknown.
        """
        if w == '###':
            if t == '###':
                return 1
            else:
                return 0
        # if w not in self.vocabulary:
        #     return lamb / (self.t_counts[t] + lamb * self.vocab_size)
        if (w, t) in self.distribution:
            return self.distribution[w, t]
        self.distribution[w, t] = (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)
        return self.distribution[w, t]


class PwtUnknown:
    """
    Computes the initial distribution of p(w|t).
    The distribution of the unknown words is estimated from the words that occur only once in the data.
    """
    wt_counts = []
    t_counts = []
    vocab_size = 0

    def __init__(self, data, wordset_size, tagset_size, hard_zeros=True):
        self.t_counts = Counter([t for (_, t) in data])
        self.w_counts = Counter([w for (w, _) in data])
        # guesstimate the probabilities
        self.wt_counts = Counter([(w, t) if self.w_counts[w] > 1 else ('@UNK', t) for w, t in data])
        # remove the words that are seen only once from the vocabulary and replace them with the default @UNK token
        for k in list(self.w_counts):
            if self.w_counts[k] < 2:
                del self.w_counts[k]
        self.vocabulary = self.w_counts.keys()
        self.vocab_size = wordset_size * tagset_size
        self.tagset_len = tagset_size
        self.hard_zeros = hard_zeros
        # this distribution can be accessed from the outside
        self.distribution = defaultdict(lambda: 0)

    def p(self, w, t):
        """
        Returns the p(w|t), smoothed by adding less than 1.
        """
        if w not in self.vocabulary:
            w = '@UNK'
        # if it was precomputed
        if (w, t) in self.distribution:
            return self.distribution[w, t]
        # precompute and store
        # if w not in self.vocabulary:
        #     return (self.wt_counts['@UNK', t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)
        # if self.wt_counts[w, t] == 0 and self.hard_zeros:
        #     return 0
        # return (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)
        self.distribution[w, t] = self.wt_counts[w, t] / self.t_counts[t]
        return self.distribution[w, t]


class PwtUnknownSmooth(PwtUnknown):
    def p(self, w, t, lamb=2 ** (-10)):
        """
        Returns the p(w|t), smoothed by adding less than 1.
        """
        if w == '###':
            if t == '###':
                return 1
            else:
                return 0
        if w not in self.vocabulary:
            w = '@UNK'
        # if it was precomputed
        if (w, t) in self.distribution:
            return self.distribution[w, t]
        # precompute, store and return
        self.distribution[w, t] = (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)
        return self.distribution[w, t]


class Ptt:
    """
    Class for getting smoothed arc probability.  Do linear interpolation trigram smoothing, need estimated probabilities
    form train data, heldout data fo estimate lambdas and testing data.
    """

    def __init__(self, ps, heldout, train, less_memory=False):
        self.memory = less_memory
        # compute lambdas
        self.l = self.EMalgorithm(heldout, ps)
        print("INFO: [l0, l1, l2, l3] =", self.l)
        # if not less_memory:
        #     self.p_t = self.compute_full(ps, heldout, train)
        # self.p_t_known = self.compute_known(ps, heldout, train)
        self.ps = ps
        self.pl0 = self.ps[0] * self.l[0]
        # for i in range(1, 4):
        #     self.pl[i] = Counter()
        #     for t in ps[i]:
        #         self.pl[i][t] = self.ps[i][t] * self.l[i]
        self.distribution = Counter()

    # def get_ptt(self, t1, t2, t3):
    #     """
    #     Returns smoothed p(t3|t1,t2).
    #     """
    #     if self.memory:
    #         return self.pl0 + self.pl[1][t3] + self.pl[2][(t2, t3)] + self.pl[3][(t1, t2, t3)]
    #     else:
    #         return self.p_t[t1, t2, t3]

    def p(self, t1, t2, t3):
        if (t1, t2, t3) in self.distribution:
            return self.distribution[t1, t2, t3]
        self.distribution[t1, t2, t3] = self.pl0 + self.l[1] * self.ps[1][t3] + self.l[2] * self.ps[2][t2, t3] + \
                                        self.l[3] * self.ps[3][t1, t2, t3]
        return self.distribution[t1, t2, t3]

    # def get_ptt_nonzero(self, t1, t2, t3):
    #     """
    #     Returns linear interpolated (t3|t1,t2), use only for known t1,t2,t3!
    #     """
    #     return self.p_t_known[t1, t2, t3]
    #
    # def compute_full(self, p, heldout, traindata):
    #
    #     ttrainset = set(traindata)
    #     pt_em = {
    #         (i, j, k): (
    #             self.l[0] * p[0] + self.l[1] * p[1][k] + self.l[2] * p[2][(j, k)] + self.l[
    #                 3] * p[3][(i, j, k)])
    #         for i in ttrainset for j in ttrainset for k in ttrainset}
    #     return pt_em
    #
    # def compute_known(self, p, heldout, traindata):
    #
    #     triset = set([i for i in zip(traindata[:-2], traindata[1:-1], traindata[2:])])
    #     ttrainset = set(traindata)
    #     pt_em = {
    #         (i, j, k): (
    #             self.l[0] * p[0] + self.l[1] * p[1][k] + self.l[2] * p[2][(j, k)] + self.l[
    #                 3] * p[3][(i, j, k)])
    #         for (i, j, k) in triset}
    #     return pt_em

    def EMiter(self, data, p, l):
        """One iteration of EM algorithm."""
        tri = [u for u in zip(data[:-2], data[1:-1], data[2:])]
        pp = {
            (i, j, k): l[3] * p[3][(i, j, k)] + l[2] * p[2][(j, k)] + l[1] * p[1][k] + l[0] * p[0]
            for (i, j, k) in set(tri)
            }  # new p'(lambda)
        c = [0, 0, 0, 0]
        for (i, j, k) in tri:
            pptemp = pp[(i, j, k)]
            c[0] += l[0] * p[0] / pptemp
            c[1] += l[1] * p[1][k] / pptemp
            c[2] += l[2] * p[2][(j, k)] / pptemp
            c[3] += l[3] * p[3][(i, j, k)] / pptemp
        # print(i, j, k, ':', c)
        return [i / sum(c) for i in c]  # normalised

    def EMalgorithm(self, data, p, e=0.001):  # heldoutdata
        """EM algorithm, input: data: heldout data, p: probabilities counted form training data """
        l = [10, 10, 10, 10]  # infinity, due to first while
        nextl = [0.25, 0.25, 0.25, 0.25]  # lambdas counted in the next iteration
        itercount = 0
        while (abs(l[0] - nextl[0]) >= e or abs(l[1] - nextl[1]) >= e or abs(l[2] - nextl[2]) >= e or abs(
                    l[3] - nextl[3]) >= e):  # expected precision is not yet achieved
            l = nextl
            nextl = self.EMiter(data, p, l)
            itercount = itercount + 1
        print("INFO: EM converged in", str(itercount), "iterations with convergence criterion", str(e))
        return nextl


class PttModified:
    def __init__(self, initial_distribution, state_set, possible_next):
        self.init_disrt = initial_distribution
        self.dist = defaultdict(lambda: 0)
        self.smooth_distribution(state_set, possible_next)

    def smooth_distribution(self, state_set, possible_next, lamb=2 ** (-10)):
        for state in state_set:
            suma = 0
            # add lambda to every transition and then normalize it
            for v in possible_next[state[1]]:
                suma += self.init_disrt[state[0], state[1], v] + lamb
            # normalize all the transitions
            for v in possible_next[state[1]]:
                self.dist[state[0], state[1], v] = (self.init_disrt[state[0], state[1], v] + lamb) / suma

    def p(self, u, v, w):
        # this last transition is just a fake transition so that we don't have to
        # compute maximum
        # we have precomputed all the possible transitions
        return self.dist[u, v, w]


class PwtModified:
    def __init__(self, initial_distribution, state_counts, vocab_size):
        self.vocab_size = vocab_size
        self.init_disrt = initial_distribution
        self.state_counts = state_counts
        self.dist = defaultdict(lambda: 0)

    def p(self, word, state, lamb=2 ** (-10)):
        if (word, state) in self.dist:
            return self.dist[word, state]
        else:
            # precompute the value and store it
            qqq = (self.init_disrt[word, state] * self.state_counts[state] + lamb) / (
                self.state_counts[state] + lamb * self.vocab_size)
            self.dist[word, state] = qqq
            return qqq
