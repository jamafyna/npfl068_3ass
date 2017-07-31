import numpy as np
from collections import Counter, defaultdict
from classes import Linear


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

    # tag unigram probabilities
    p1 = {t: tags_uniq[t] / count for t in tags_uniq}

    # bigram probabilities
    bigram_tags = Counter(zip(tags[:-1], tags[1:]))
    p2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}

    return p_output, (p0, p1, p2)


def compute_forward_probabilities(transition_matrix, emission_matrix, sentence, n):
    k = sentence.size
    alpha = np.zeros((n, k + 1))
    # forward computation
    alpha[:, 0] = 1.0 / n
    for obs_ind in range(k):
        # we select the "column" in the complete trellis
        trellis_column = np.matrix(alpha[:, obs_ind])
        alpha[:, obs_ind + 1] = trellis_column * np.matrix(transition_matrix) * \
                                np.matrix(np.diag(emission_matrix[:, sentence[obs_ind]]))
    return alpha


def compute_backward_probabilities(transition_matrix, emission_matrix, sentence, n):
    k = sentence.size
    beta = np.zeros((n, k + 1))
    # backward computation
    beta[:, -1] = 1.0
    for obs_ind in range(k, 0, -1):
        trellis_column = np.matrix(beta[:, obs_ind]).transpose()
        beta[:, obs_ind - 1] = (np.matrix(transition_matrix) * \
                                np.matrix(np.diag(emission_matrix[:, sentence[obs_ind - 1]])) * \
                                trellis_column).transpose()
    return beta


def collect_counts():
    pass


def reestimate():
    pass


# def baum_welch(training_data, held_out_data):
#     # initialization
#     whole_data = training_data

file = open('data/texten2.ptg', 'rt')
data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))
train_data = data[:-60000]
held_out_data = data[-60000:-40000]
test_data = data[-40000:]

# we will use only bigram distribution, otherwise the number of state would be much larger
# get the words and the tags
tag_set = list(set([t for w, t in train_data]))
words = [w for w, t in train_data]
word_counts = Counter(words)
# words that are rarely in the test data are also unlikely to be in the testing data
# but it significantly reduces the size of the matrix
vocabulary = [w for w in word_counts if word_counts[w] > 5]
# create dictionaries to convert word to index
i = 1
# if the word is not present, return 0 --- treat unknown words as a special OOV token
# does not matter much, because those words are rare anyway
word_to_index = defaultdict(lambda: 0)
for word in vocabulary:
    word_to_index[word] = 1
    i += 1
# tags are the states
state_count = len(tag_set)

# get P(w|t) and n-gram distributions, use only the first 10,000 words of T to estimate the initial (raw) parameters
emission_distribution, dists = get_distributions(train_data[:10000])
# get the smoothed transition distribution
trans_distribution = Linear(held_out_data, dists[0], dists[1], dists[2])

# allocate the matrix
transition_matrix = np.zeros((state_count, state_count))
# fill in the transition matrix with the estimated probabilities
for i in range(state_count):
    for j in range(state_count):
        transition_matrix[i, j] = trans_distribution.p(tag_set[i], tag_set[j])

emission_matrix = np.zeros((state_count, len(vocabulary)))
for i in range(state_count):
    for j in range(state_count):
        transition_matrix[i, j] = trans_distribution.p(tag_set[i], tag_set[j])

convergence = False
while not convergence:
    alfa = compute_forward_probabilities(trans_distribution, output_distribution)
    beta = compute_backward_probabilities(trans_distribution, output_distribution)
    collect_counts()
    reestimate()
print('Result')
