from collections import Counter, defaultdict
import dill


def normalize_and_prune(my_counter, threshold=20):
    """
    Normalizes and prunes the given Counter
    :param my_counter: Counter to be reorganized
    :param threshold: Pruning threshold
    :return: A new pruned and normalized counter
    """
    tmp_counter = Counter()
    tmp_sum = 0
    for x, y in my_counter.most_common(threshold):
        if y > 0:
            tmp_counter[x] = y
            tmp_sum += y
    for key in tmp_counter:
        tmp_counter[key] /= tmp_sum
    return tmp_counter


def normalize(my_counter):
    """
    Normalizes a given Counter or a dictionary
    :param my_counter: The counter to be normalized
    :return: nothing, edits in place
    """
    tmp_sum = sum(my_counter.values())
    for key in my_counter:
        my_counter[key] /= tmp_sum


def compute_alpha(sentence, t, e, beta, possible_next):
    """
    Computes the forward probabilities
    :param beta: Already computed backward probabilities
    :param sentence: The training sentence
    :param t: The transition probabilities distribution
    :param e: The emission probability distribution
    :return: A dictionary of Counters representing the forward probabilities, time is the first key
    """
    alpha = defaultdict(Counter)
    # starting state
    u = '###'
    v = '###'
    alpha[1][(u, v)] = 1  # everything else is zero because it's deterministic

    # iterate over the observations
    # 2 is the first time when an actual output happens
    for time in range(2, len(sentence) - 1):
        # iterate over all the previous trellis stages
        for u, v in alpha[time - 1].keys():
            # for all the possible next states uv ~> vw
            # consider only states that survived the pruning in beta
            for w in possible_next[v]:
                if (v, w) in beta[time]:
                    # simulate transitions to w over the time-th observation
                    qqq = alpha[time - 1][u, v] * t[u, v, w] * e[sentence[time], (v, w)]
                    if qqq > 0:
                        alpha[time][v, w] += qqq
        # no need to prune, because we considered only the states that survived backward running
        # therefore there will be at most threshold number of elements
        # print('Q:', time, len(beta[time].keys()))
        normalize(alpha[time])
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
    w = '###'
    v = '###'
    for u in possible_prev[v]:
        beta[len(sentence) - 2][u, v] = 1
    normalize(beta[len(sentence) - 2])

    # iterate over the observations backwards
    for time in range(len(sentence) - 3, 0, -1):
        # iterate over all the previous trellis stages
        for v, w in beta[time + 1].keys():
            # for all the possible previous states uv <~ vw
            for u in possible_prev[v]:
                # simulate transitions to w over the time-th observation
                qqq = beta[time + 1][v, w] * t[u, v, w] * e[sentence[time], (u, v)]
                if qqq > 0:  # prevents adding zeros to the dictionary and iterating over them
                    beta[time][u, v] += qqq
        # try:
        # print('T:', time, len(beta[time].keys()))
        normalize(beta[time])
        # except:
        #     print(time, sum(beta[time].values()), type(beta[time]))
    # a trick, because we use determinstic starting state, we can exclude most of the states
    # time = 2
    # for v, w in beta[time + 1].keys():
    #     # for all the possible previous states uv <~ vw
    #     u = '###'
    #     # simulate transitions to w over the time-th observation
    #     beta[time][u, v] += beta[time + 1][v, w] * t[u, v, w] * e[sentence[time], v]
    # normalize(beta[time])
    # beta[1]['###', '###'] = 1
    return beta


def baum_welch(transition_p, emission_p, training_data, graph_forward, graph_backward, states, eps=0.01,
               file='text.txt', fold=0):
    """
    Gets the training data splitted into sequnces (sentences) and modifies transition and emission probability
     distributions until convergence is reached.
     For detailed explanation of the algorithm see Rabiner's paper (1989)
    :param transition_p: The transition probability distribution, a dictionary returning floats for p[t1, t2, t3]
    :param emission_p:  The emission probability distribution p(w|t), a dictionary returning float for p[w, t]
    :param training_data: The training data split into sentences
    :param graph_forward: A dictionary of bigram of tags, key is the first tag in the bigram
    :param graph_backward: A dictionary of bigram of tags, key is the last tag in the bigram
    :return: A dictionary of counters, first level is by time, second by state
    """
    # get the number of training sequences/sentences
    n = len(training_data)
    # alpha(state, time)
    # alpha = Counter()
    # beta(state, time)
    # beta = Counter()
    # c(o,s1,s2) or the numerator in (111) in Rabiner
    # c = Counter()
    # probability of being in s at any time
    # sum of alpha(s, i) * beta(s, i) over time
    # denominator in Rabiner (111)
    # d = Counter()
    # probability of being in s which emits obs at any time
    # numerator in Rabiner (110)
    # e = Counter()

    num_iter = 0

    # "Expectation" step

    convergence = False
    while not convergence:
        num_iter += 1
        print('INFO:', num_iter, '-th iteration')
        c = defaultdict(lambda: 0)
        d = defaultdict(lambda: 0)
        e = defaultdict(lambda: 0)
        for sentence in training_data[:1]:
            T = len(sentence)
            # compute forward and backward probabilities
            # try:
            beta = compute_beta(sentence, transition_p, emission_p, graph_backward)

            alpha = compute_alpha(sentence, transition_p, emission_p, beta, graph_forward)

            # sum over time
            for t in range(1, T - 1):
                for u, v in alpha[t].keys():
                    # consider only the transitions observed in the data old -> new
                    for w in graph_forward[v]:
                        c[u, v, w] += alpha[t][u, v] * transition_p[u, v, w] * \
                                      emission_p[sentence[t + 1], (v, w)] * beta[t + 1][v, w]

            # compute d
            # probability of being in state at any given time
            for state in states:
                for t in range(1, T - 1):
                    qqq = alpha[t][state] * beta[t][state]
                    if qqq > 0:  # preventing adding zero counts that cause trouble later
                        # can be also avoided by using Counter, but it's much slower
                        d[state] += qqq

            # compute e
            # probability of being in state at any given time and emitting the desired observation
            for state in states:
                for t in range(1, T - 1):
                    obs = sentence[t]
                    qqq = alpha[t][state] * beta[t][state]
                    if qqq > 0:
                        e[obs, state] += qqq

        # Maximization step
        print('Maximization')
        # new dictionaries
        new_tran = defaultdict(lambda: 0)
        new_emis = defaultdict(lambda: 0)

        # check convergence criterion
        conv = True

        # normalize over the number of observations
        # use MLE for sentence distribution
        for dist in [c, d, e]:
            for key in dist:
                dist[key] /= n

        # re-estimate the forward probabilities
        for i, j, k in c.keys():
            if d[i, j] > 0:  # this should be always true anyway
                new_tran[i, j, k] = c[i, j, k] / d[i, j]
                # it takes only one value to not change enough
                if abs(transition_p[i, j, k] - new_tran[i, j, k]) > eps:
                    conv = False

        # re-estimate emission probabilities
        for o, s in e.keys():
            if d[s] > 0:  # this shold be always true anyway
                new_emis[o, s] = e[o, s] / d[s]
                if abs(new_emis[o, s] - emission_p[o, s]) > eps:
                    conv = False

        # remember the new set of parameters
        emission_p = new_emis
        transition_p = new_tran
        convergence = conv

        file_name = file + '.' + str(fold) + '.it' + str(num_iter)
        dill.dump(emission_p, open(file_name + 'pwt.p', 'wb'))
        dill.dump(transition_p, open(file_name + 'ptt.p', 'wb'))

    print('INFO: Baum-Welch converged in', num_iter, 'iterations')
    return emission_p, transition_p
