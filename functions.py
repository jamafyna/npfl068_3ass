import numpy as np
from collections import Counter, defaultdict
from classes import Linear, LexicalDistribution


def get_initial_parameters(tags):
    """
    Computes uniform, unigram, bigram and trigram distributions from given data.
    """
    tags_uniq = Counter(tags)
    # constant probability mass (avoiding zeros)
    p_t0 = 1 / len(tags_uniq)
    # unigram probabilities
    p_t1 = Counter(tags)
    # bigram probabilities
    bigram_tags = Counter(zip(tags, tags[1:]))
    p_t2 = Counter(bigram_tags)
    # p_t2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}
    # trigram probabilities
    trigram_tags = Counter([trig for trig in zip(tags, tags[1:-1], tags[2:])])
    p_t3 = Counter(trigram_tags)
    # p_t3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}
    for key in p_t3:
        p_t3[key] /= p_t2[key[:2]]
    for key in p_t2:
        p_t2[key] /= p_t1[key[0]]
    for key in p_t1:
        p_t1[key] /= len(tags)
    p_tt = [p_t0, p_t1, p_t2, p_t3]
    return p_tt, set(trigram_tags)


# todo: remove this function
def get_distributions(data_wt):
    # get the length of data
    count = len(data_wt)
    # get just the tags from the data
    tags = [t for (_, t) in data_wt]
    # tag unigram counts
    tags_uniq = Counter(tags)
    # output probabilities distribution --- P(w|t)
    # data_uniq = Counter(data_wt)
    # p_output = {(w, t): data_uniq[w, t] / tags_uniq[t] for (w, t) in data_uniq}
    p_output = LexicalDistribution(data_wt)

    # constant probability mass (avoiding zeros)
    p0 = 1 / len(tags_uniq)

    # tag unigram probabilities
    p1 = {t: tags_uniq[t] / count for t in tags_uniq}

    # bigram probabilities
    bigram_tags = Counter(zip(tags[:-1], tags[1:]))
    p2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}

    return p_output, (p0, p1, p2)


def preprocess_data(my_data, word_to_index):
    """Splits the data into sentences, strips off the tags and converts them to indices"""
    processed_data = []
    s = []
    for my_word, tag in my_data:
        # the first token is not really an observation
        if my_word != '###':
            s.append(word_to_index[my_word])
        else:
            if s:
                processed_data.append(s)
            s = []
    if s:
        processed_data.append(s)
    return processed_data


def compute_forward_probabilities(t_matrix, e_matrix, sentence, n):
    """Computes the forward probability masses"""
    l = len(sentence)
    alpha_m = np.zeros((n, l + 1))
    # the starting state is 0
    # at the beginning, just the probability distribution of states follow by the starting state
    alpha_m[:, 0] = t_matrix[0, :]
    # for all the observations -- words in the sentence
    for ii in range(l):
        # we select the "column" in the complete trellis
        trellis_c = np.matrix(alpha_m[:, ii])
        #      Alpha(s',i) = alpha(s,i-1) * p(s'|s) * p(y_i|s)
        alpha_m[:, ii + 1] = trellis_c * t_matrix * np.diag(e_matrix[:, sentence[ii]])
        # normalization after each stage (slide 185)
        alpha_m[:, ii + 1] /= np.sum(alpha_m[:, ii + 1])
    return alpha_m


def compute_backward_probabilities(t_matrix, e_matrix, sentence, n):
    """Computes the backward probability masses"""
    l = len(sentence)
    beta_m = np.zeros((n, l + 1))
    beta_m[:, -1] = 1.0
    for ii in range(l, 0, -1):
        trellis_c = np.matrix(beta_m[:, ii]).transpose()
        beta_m[:, ii - 1] = np.transpose(t_matrix * np.diag(e_matrix[:, sentence[ii - 1]]) * trellis_c)
        # normalization
        beta_m[:, ii - 1] /= np.sum(beta_m[:, ii - 1])
    return beta_m


def collect_counts_and_reestimate(alfa, beta, t_matrix, e_matrix, sentence, n):
    l = len(sentence)
    inc = np.zeros((n, n, l))
    for ii in range(n):
        for jj in range(n):
            for k in range(l):
                # inc(y,s,s')  = alfa(s,i)   * p(s'|s)          * p(y|s')                   * beta(s'|i+1)
                # from slide 189, in Wikipedia it is denoted with ksi
                inc[ii, jj, k] = alfa[ii, k] * t_matrix[ii, jj] * e_matrix[jj, sentence[k]] * beta[jj, k + 1]

    t_matrix_new = np.zeros(t_matrix.size)
    # probability that we are in state state s in time i -- gama[s,i]
    # see https://en.wikipedia.org/wiki/Baum–Welch_algorithm
    gamma = np.multiply(alfa, beta)
    # normalize each trellis stage i.e. normalize columns -- divide each column by its sum
    gamma /= np.sum(gamma, 0)
    # compute
    for ii in range(n):
        for jj in range(n):
            # transition probability = sum inc over time / sum gamma over time
            # to get transition s -> s' independent of time, supposing that we are in s
            t_matrix_new[ii, jj] = np.sum(inc[ii, jj, :]) / np.sum(gamma[ii, :])
    t_matrix_new /= np.sum(t_matrix_new, 1)

    # get the effective vocabulary size
    vocab_size = e_matrix.shape[1]
    e_matrix_new = np.zeros(e_matrix.shape)
    # for every state
    for ii in range(n):
        # and every known (and the special OOV word)
        for word in range(vocab_size):
            # get all the positions where the word is in this sentence
            indices = [i + 1 for i, x in enumerate(sentence) if x == word]
            # estimate the emission probability of s emitting y as
            # probability of state s emitting y in any time (sum over times)
            # over total probability of being in that state at any time (at the beginning we are in imaginary state 0)
            e_matrix_new[ii, word] = np.sum(gamma[ii, indices]) / np.sum(gamma[ii, 1:])
    # normalize rows, because they represent emittion distribution of the state
    e_matrix_new /= np.sum(e_matrix_new, 1)
    # compare
    return t_matrix_new, e_matrix_new


def baum_welch(training_data, held_out_data, epsilon=0.001):
    # initialization

    # we will use only bigram distribution, otherwise the number of state would be much larger
    # get the words and the tags
    tag_set = list(set([t for w, t in training_data]))
    # make the starting state state number 0
    tag_set.remove('###')
    tag_set = ['###'] + tag_set
    words = [w for w, t in training_data]
    word_counts = Counter(words)
    # words that are rarely in the test data are also unlikely to be in the testing data
    # but it significantly reduces the size of the matrix
    vocabulary = [w for w in word_counts if word_counts[w] > 1]
    vocabulary.remove('###')
    # make the starting state 0
    vocabulary = ['###'] + vocabulary
    # create dictionaries to convert word to index
    # if the word is not present, return maximum --- treat unknown words as a special OOV token
    # does not matter much, because those words are rare anyway
    word_to_index = defaultdict(lambda: len(vocabulary))
    for i in range(len(vocabulary)):
        word_to_index[vocabulary[i]] = i

    # tags are the states
    state_count = len(tag_set)

    # get P(w|t) and n-gram distributions, use only the first 10,000 words of T to estimate the initial (raw) parameters
    emission_distribution, dists = get_distributions(training_data[:10000])
    # get the smoothed transition distribution
    trans_distribution = Linear(held_out_data, dists[0], dists[1], dists[2])

    # allocate the matrix
    transition_matrix = np.zeros((state_count, state_count))
    # fill in the transition matrix with the estimated probabilities
    for i in range(state_count):
        for j in range(state_count):
            transition_matrix[i, j] = trans_distribution.p(tag_set[i], tag_set[j])

    emission_matrix = np.zeros((state_count, len(vocabulary) + 1))
    for i in range(state_count):
        for j in range(len(vocabulary)):
            emission_matrix[i, j] = emission_distribution.p(vocabulary[j], tag_set[i])

    # preprocessed the data
    data_train = preprocess_data(training_data, word_to_index)

    convergence = False
    while not convergence:
        for training_sentence in data_train:
            print('DEBUG: Processing a sentence')
            alfa = compute_forward_probabilities(transition_matrix, emission_matrix, training_sentence, state_count)
            beta = compute_backward_probabilities(transition_matrix, emission_matrix, training_sentence, state_count)
            t_new, e_new = collect_counts_and_reestimate(alfa, beta, transition_matrix, emission_matrix,
                                                         training_sentence, state_count)
            # check convergence criterion
            if np.linalg.norm(transition_matrix - t_new) < epsilon and np.linalg.norm(
                            emission_matrix - e_new) < epsilon:
                convergence = True
            transition_matrix = t_new
            emission_matrix = e_new

    return transition_matrix, emission_matrix
