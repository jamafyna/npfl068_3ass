from collections import Counter, defaultdict


def compute_alpha(sentence, t, e, possible_next):
    """
    Computes the forward probabilities
    :param sentence: The training sentence
    :param t: The transition probabilities distribution
    :param e: The emission probability distribution
    :param possible_next: A dictionary returning an iterable over next possible tags
    :return:
    """
    alpha = defaultdict(Counter)
    # starting state
    u = '###'
    v = '###'
    alpha[1][(u, v)] = 1  # everything else is zero because it's deterministic

    # iterate over the observations
    # 2 is the first where an actual output happens
    for time in range(2, len(sentence)):
        # iterate over all the previous trellis stages
        for u, v in alpha[time - 1].keys():
            # for all the possible next states uv ~> vw
            # for the sake of time and memory, consider only states from the training data
            for w in possible_next[v]:
                # simulate transitions to w over the time-th observation
                alpha[time][u, v] += alpha[time - 1][u, v] * t[u, v, w] * e[sentence[time][0], w]

    return alpha


def compute_beta(sentence, t, e, possible_prev):
    """
    Computes the forward probabilities
    :param sentence: The training sentence
    :param t: The transition probabilities distribution
    :param e: The emission probability distribution
    :param possible_prev: A dictionary returning an iterable over next possible tags
    :return: A dictionary of counters, first level is by time, the second by state
    """
    beta = defaultdict(Counter)
    # ending state is also deterministic
    u = '~~~'
    v = '~~~'
    beta[len(sentence) - 1][(u, v)] = 1  # everything else is zero because it's deterministic

    # iterate over the observations backwards
    for time in range(len(sentence) - 1, 0, -1):
        # iterate over all the previous trellis stages
        for u, v in beta[time + 1].keys():
            # for all the possible previous states uv <~ vw
            for w in possible_prev[u]:
                # simulate transitions to w over the time-th observation
                beta[time][u, v] += beta[time + 1][u, v] * t[u, v, w] * e[sentence[time][0], w]

    return beta


def baum_welch(transition_p, emission_p, training_data, graph_forward, graph_backward, states):
    """
    Gets the training data splitted into sequnces (sentences) and modifies transition and emission probability
     distributions until convergence is reached.
     For detailed explanation of the algorithm see Rabiner's paper (1989)
    :param transition_p: The transition probability distribution, a dictionary returning floats for p[t1, t2, t3]
    :param emission_p:  The emission probability distribution p(w|t), a dictionary returning float for p[w, t]
    :param training_data: The training data split into sentences
    :return: A dictionary of counters, first level is by time, second by state
    """
    # get the number of training sequences
    n = len(training_data)
    # alpha(state, time)
    alpha = Counter()
    # beta(state, time)
    beta = Counter()
    # c(o,s1,s2) or the numerator in (111) in Rabiner
    c = Counter()
    # probability of being in s at any time
    # sum of alpha(s, i) * beta(s, i) over time
    # denominator in Rabiner (111)
    d = Counter()
    # probability of being in s which emits obs at any time
    # numerator in Rabiner (110)
    e = Counter()

    # "Expectation" step

    for sentence in training_data:
        T = len(sentence)
        # compute forward and backward probabilities
        alpha = compute_alpha(transition_p, emission_p)
        beta = compute_beta(transition_p, emission_p)

        for t in range(T):
            # todo rozmysli si to
            for old_state in alpha.keys():
                # consider all the possible transitions old -> new
                for new_state in graph_forward[old_state]:
                    c[old_state, new_state] += alpha[old_state, t] * transition_p.p(old_state, new_state) * \
                                               emission_p.p(new_state, sentence[t]) * beta[new_state, t + 1]

        # compute d
        # probability of being in state at any given time
        for state in states:
            for t in range(T):
                d[state] += alpha[state, t] + beta[state, t]

        # compute e
        # probability of being in state at any given time and emitting the desired observation
        for state in states:
            # todo: get observations
            for obs in ['a', 'b']:
                for t in range(T):
                    e[state, obs] += alpha[state, t] + beta[state, t]

    # Maximization step

    # normalize over the number of observations
    # use MLE for sentence distribution
    for dist in [c, d, e]:
        for key in alpha:
            dist[key] /= n

    # reestimate the forward probabilities
    for i, j in transition_p.keys():
        transition_p[i, j] = c[i, j] / d[i]

    # reestimate emission probabilities
    for w, i in emission_p.keys():
        emission_p[w, i] = e[w, i] / d[i]
