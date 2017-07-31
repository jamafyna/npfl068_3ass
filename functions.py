import numpy as np
from collections import Counter
from classes import LinearSmoothedDistribution


def get_distributions(data_wt):
    # get the length of data
    count = len(data_wt)
    # get just the tags from the data
    tags = [t for (_, t) in data_wt]
    # tag unigram counts
    tags_uniq = Counter(tags)
    data_uniq = Counter(data_wt)
    # output probabilities distribution --- P(w|t)
    p_output = {(w, t): data_uniq[w, t] / tags_uniq[t] for (w, t) in data_uniq}

    # constant probability mass (avoiding zeros)
    p0 = 1 / len(tags_uniq)

    # unigram probabilities
    p1 = {t: tags_uniq[t] / count for t in tags_uniq}

    # bigram probabilities
    bigram_tags = Counter(zip(tags[:-1], tags[1:]))
    p2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}

    # trigram probabilities
    trigram_tags = Counter([trig for trig in zip(tags[:-2], tags[1:-1], tags[2:])])
    p3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}
    return p_output, (p0, p1, p2, p3)


def compute_forward_probabilities(transition_matrix, emission_matrix, sentence, n):
    k = sentence.size
    alpha = np.zeros((n, k + 1))
    # forward computation
    alpha[:, 0] = 1.0 / n
    for obs_ind in range(k):
        # we select the "column" in the complete trellis
        f_row_vec = np.matrix(alpha[:, obs_ind])
        alpha[:, obs_ind + 1] = f_row_vec * np.matrix(transition_matrix) * \
                                np.matrix(np.diag(emission_matrix[:, sentence[obs_ind]]))
    return alpha


def compute_backward_probabilities(transition_matrix, emission_matrix, sentence, n):
    k = sentence.size
    beta = np.zeros((n, k + 1))
    # backward computation
    beta[:, -1] = 1.0
    for obs_ind in range(k, 0, -1):
        b_col_vec = np.matrix(beta[:, obs_ind]).transpose()
        beta[:, obs_ind - 1] = (np.matrix(transition_matrix) * \
                                np.matrix(np.diag(emission_matrix[:, sentence[obs_ind - 1]])) * \
                                b_col_vec).transpose()
    return beta


def collect_counts():
    pass


def reestimate():
    pass


def baum_welch(training_data, held_out_data):
    # initialization
    output_distribution, dists = get_distributions(training_data[:10000])
    # get the smoothed transition distribution
    trans_distribution = LinearSmoothedDistribution(held_out_data, dists[0], dists[1], dists[2], dists[3])
    convergence = False
    while not convergence:
        alfa = compute_forward_probabilities(trans_distribution, output_distribution)
        beta = compute_backward_probabilities(trans_distribution, output_distribution)
        collect_counts()
        reestimate()
    print('Result')
