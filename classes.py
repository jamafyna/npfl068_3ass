from collections import Counter


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

    def p(self, t1, t2, t3):
        """Returns the probability of a trigram under the current model"""
        return self.lambdas[3] * self.p_3[t1, t2, t3] + self.lambdas[2] * self.p_2[t2, t3] + \
               self.lambdas[1] * self.p_1[t3] + self.lambdas[0] * self.p_0

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


class LexicalDistribution:
    """
    Computes the initial distribution of p(w|t) and smooths it with add less than one smoothing
    """

    def __init__(self, data):
        self.t_counts = Counter([t for (_, t) in data])
        self.w_counts = Counter([w for w, t in data])
        self.wt_counts = Counter(data)
        self.vocab_size = len(self.w_counts.keys())
        self.tagset_len = len(self.t_counts.keys())

    def p(self, w, t, lamb=2 ** (-10)):
        """
        Returns smoothed p(w|t)
        """
        # todo: Vyrobiť rozdelenie, ktoré preferuje pre neznáme slová tagy, ktoré sú časté so zriedkavými slovami.
        return (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.tagset_len)


class Pwt:
    """
    Computes the initial distribution of p(w|t) and smooths it.
    """
    wt_counts = []
    t_counts = []
    vocab_size = 0

    def __init__(self, data, wordset_size, tagset_size):
        self.t_counts = Counter([t for (_, t) in data])
        self.wt_counts = Counter(data)  # Counter([(w, t) for (w, t) in data_wt])
        self.vocab_size = wordset_size * tagset_size
        self.tagset_len = tagset_size

    def get_smoothed_pwt(self, w, t, lamb=2 ** (-10)):
        """
        Returns smoothed p(w|t), by smoothing less than 1. Suppose that w and t are from known wordset and tagset, not unknown.
        """
        if w == '~~~':
            if t == '~~~':
                return 1
            else:
                return 0
        if w == '###':
            if t == '###':
                return 1
            else:
                return 0
        return (self.wt_counts[w, t] + lamb) / (self.t_counts[t] + lamb * self.vocab_size)
