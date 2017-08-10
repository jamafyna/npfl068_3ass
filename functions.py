import warnings
import numpy as np
from collections import Counter, defaultdict
from classes import Pwt as LexicalDistribution

STARTw = "###"  # start token/token which split sentences
STARTt = "###"  # STARTw for words, STARTt for tags


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
    # iterlist = list(bigram_tags.keys())
    # if ('~~~', '###') in iterlist:
    #     bigram_tags.pop(('~~~', '###'))
    p_t2 = Counter(bigram_tags)
    # p_t2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}
    # trigram probabilities
    trigram_tags = Counter([trig for trig in zip(tags, tags[1:-1], tags[2:])])
    # if ('~~~', '###', '###') in trigram_tags.keys():
    #     trigram_tags.pop(('~~~', '###', '###'))
    # if ('~~~', '~~~', '###') in trigram_tags.keys():
    #     trigram_tags.pop(('~~~', '~~~', '###'))
    p_t3 = Counter(trigram_tags)
    # p_t3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}
    for key in p_t3:
        p_t3[key] /= p_t2[key[:2]]
    for key in p_t2:
        p_t2[key] /= p_t1[key[0]]
    for key in p_t1:
        p_t1[key] /= len(tags)
    return [p_t0, p_t1, p_t2, p_t3], set(trigram_tags)


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
            # probability of state s emitting y at any time (sum over times)
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


def fix_sentence_boundaries(data):
    # remove leading and final divisions
    while data[0] == ('###', '###'):
        data.pop(0)
    while data[-1] == ('###', '###'):
        data.pop()
    fixed_data = []
    sentence = []
    for e in data:
        if e != ('###', '###'):
            sentence.append(e)
        else:
            fixed_data.append([('###', '###'), ('###', '###')] + sentence + [('###', '###'), ('###', '###')])
            sentence = []
    if sentence:
        fixed_data.append([('###', '###'), ('###', '###')] + sentence + [('###', '###'), ('###', '###')])
    return fixed_data


def fix_sentence_boundaries_words(data):
    # remove leading and final divisions
    while data[0] == '###':
        data.pop(0)
    while data[-1] == '###':
        data.pop()
    fixed_data = []
    sentence = []
    for e in data:
        if e != '###':
            sentence.append(e)
        else:
            fixed_data.append(['###', '###'] + sentence + ['###', '###'])
            sentence = []
    if sentence:
        fixed_data.append(['###', '###'] + sentence + ['###', '###'])
    return fixed_data


def viterbi(text, tagset, wordset, trigramtagset, Pwt, Ptt, start, usetrigram=True):
    """
    Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset),
    and first half of a start state (usually end of previous sentence or start token).
    """
    if len(text) == 0:
        return []

    isOOV = False  # indicates if proceeded word is out-of-vocabulary

    if text[0] == STARTw:
        warnings.warn("inconsistend data, start tokens", Warning)
    else:
        text = [(STARTw, STARTt)] + text
    V = {}  # structure for remember Viterbi computation
    # V[time][state]=(probability,path) where state==(tag1,tag2)
    path = {}  # the best path to some state
    V[0] = {}
    V[1] = {}
    OOVcount = 0

    # --- initialisation, starting state
    V[0][start] = (0, [start])
    V[1][STARTt, STARTt] = (1, [start, (STARTw, STARTt)])
    now = 1  # value depends on k which starts from 1
    prev = 0
    if usetrigram:
        get_ptt_f = Ptt.get_ptt_nonzero
    else:
        get_ptt_f = Ptt.get_ptt
    # --- finding the best way
    for k in range(2, len(text) + 1):
        isOOV = False
        w = text[k - 1]
        now = k % 2  # k modulo 2 instead of k; it is sufficient to remember
        prev = (now + 1) % 2  # instead of k-1;          only actual and previous time
        V[now] = {}
        # count the number of OOV words
        if w not in wordset:
            OOVcount += 1
            isOOV = True
        # for all possible following tags
        for t in tagset:
            if t == STARTt:
                continue

            for (i, j) in V[prev]:
                # todo: spped up
                if usetrigram and ((i, j, t) not in trigramtagset):
                    continue

                value = V[prev][i, j][0] * get_ptt_f(i, j, t)
                # value = V[prev][i, j][0] * transition_p.p(i, j, t)
                if ((j, t) not in V[now]) or value > V[now][j, t][0]:
                    V[now][j, t] = (value * Pwt.get_smoothed_pwt(w, t, isOOV),
                                    V[prev][i, j][1] + [(w, t)])
    # --- final search the best tag sequence
    maxprob = 0
    ends = {}  # the best end state
    for s in V[now]:
        if V[now][s][0] >= maxprob:
            maxprob = V[now][s][0]
            ends = s
    if maxprob == 0 & len(text) > 1:
        warnings.warn("zero maxprobability at the end")
    return V[now][ends][1][2:], OOVcount  # only [2:] because of start tokens


def vite(sentence, tagset, emission_p, transition_p, possible_next, unknown_states=True, threshold=0, tags_dict=None):
    # maximum probability in that state in time t = 0
    alpha_t = defaultdict(lambda: 0)
    # deterministic, starting state is the only possible state
    u = '###'
    v = '###'
    alpha_t[('###', '###')] = 1
    alpha_new = Counter()  # use Counter, because defaultdict creates the element upon being touched
    # back-track pointers
    psi = defaultdict(lambda: ('###', '###'))
    psi[(1, v)] = u
    if unknown_states:
        iteration_set = tagset
    # iterate over the observations
    for time in range(2, len(sentence)):
        # iterate over all the previous trellis stage
        for u, v in alpha_t.keys():
            if alpha_t[u, v] > threshold:
                # consider all the possible next tags
                if not unknown_states:
                    iteration_set = possible_next[v]
                if tags_dict:
                    iteration_set = tags_dict[sentence[time][0]]
                for w in iteration_set:
                    # simulate transitions to w over the k-th observation
                    q = alpha_t[u, v] * transition_p.p(u, v, w) * emission_p.p(sentence[time][0], w)
                    # if a better alpha to the state (v, w) from the previous trellis stage, remember the better one
                    if (v, w) not in alpha_new:
                        alpha_new[v, w] = q
                        psi[(time, (v, w))] = (u, v)
                    elif q > alpha_new[v, w]:
                        alpha_new[v, w] = q
                        psi[(time, (v, w))] = (u, v)
        # next trellis stage completly generated, now forget the old one
        alpha_t = alpha_new
        alpha_new = defaultdict(lambda: 0)
    last = ('###', '###')
    tagged = [last[0], last[0]]
    for i in range(len(sentence) - 1, 1, -1):
        last = psi[i, last]
        # print(last)
        tagged.append(last[0])
    # print('---')
    tagged.reverse()
    return tagged


def viterbi_prunned(sentence, tagset, emission_p, transition_p, possible_next, unknown_states=True, threshold=10,
                    tags_dict=None):
    # maximum probability in that state in time t = 0
    alpha_t = Counter()
    # deterministic, starting state is the only possible state
    u = '###'
    v = '###'
    alpha_t[('###', '###')] = 1
    alpha_new = Counter()  # use Counter, because defaultdict creates the element upon being touched
    # back-track pointers
    psi = defaultdict(lambda: ('###', '###'))
    psi[(1, v)] = u
    if unknown_states:
        iteration_set = tagset
    # iterate over the observations
    for time in range(2, len(sentence)):
        # iterate over all the previous trellis stage
        for ((u, v), al) in alpha_t.most_common(threshold):
            if al > 0:
                # consider all the possible next tags
                if not unknown_states:
                    iteration_set = possible_next[v]
                if tags_dict:
                    iteration_set = tags_dict[sentence[time][0]]
                for w in iteration_set:
                    # simulate transitions to w over the k-th observation
                    try:
                        q = alpha_t[u, v] * transition_p.p(u, v, w) * emission_p.p(sentence[time][0], w)
                    except:
                        print('@', sentence[time][0], w)
                    # if a better alpha to the state (v, w) from the previous trellis stage, remember the better one
                    if q > alpha_new[v, w]:
                        alpha_new[v, w] = q
                        psi[(time, (v, w))] = (u, v)
        # next trellis stage completly generated, now forget the old one
        alpha_t = alpha_new
        alpha_new = Counter()
    last = ('###', '###')
    tagged = [last[0], last[0]]
    for i in range(len(sentence) - 1, 1, -1):
        last = psi[i, last]
        # print(last)
        tagged.append(last[0])
    # print('---')
    tagged.reverse()
    return tagged


def viterbi_prunned_modified(sentence, tagset, emission_p, transition_p, possible_next, unknown_states=True,
                             threshold=10,
                             tags_dict=None):
    # maximum probability in that state in time t = 0
    alpha_t = Counter()
    # deterministic, starting state is the only possible state
    u = '###'
    v = '###'
    alpha_t[('###', '###')] = 1
    alpha_new = Counter()  # use Counter, because defaultdict creates the element upon being touched
    # back-track pointers
    psi = defaultdict(lambda: ('###', '###'))
    psi[(1, v)] = u
    if unknown_states:
        iteration_set = tagset
    # iterate over the observations
    for time in range(2, len(sentence) - 3):
        # iterate over all the previous trellis stage
        for ((u, v), al) in alpha_t.most_common(threshold):
            if al > 0:
                # consider all the possible next tags
                if not unknown_states:
                    iteration_set = possible_next[v]
                if tags_dict:
                    iteration_set = tags_dict[sentence[time][0]]
                for w in iteration_set:
                    # simulate transitions to w over the k-th observation
                    try:
                        q = alpha_t[u, v] * transition_p.p(u, v, w) * emission_p.p(sentence[time][0], (v, w))
                    except:
                        print('@', sentence[time][0], w)
                    # if a better alpha to the state (v, w) from the previous trellis stage, remember the better one
                    if q > alpha_new[v, w]:
                        alpha_new[v, w] = q
                        psi[(time, (v, w))] = (u, v)
        # next trellis stage completly generated, now forget the old one
        alpha_t = alpha_new
        alpha_new = Counter()
    last = alpha_t.most_common(1)[0][0]  # ('###', '###')
    tagged = ['###', '###']
    for i in range(len(sentence) - 1, 1, -1):
        last = psi[i, last]
        # print(last)
        tagged.append(last[0])
    # print('---')
    tagged.reverse()
    return tagged


def evaluate_test_data(data_S, tagsetT, pwt, ptt, possible_next_tags, threshold=20, unk=False, possible_tags=None,
                       modified=False):
    total = 0
    correct = 0
    if threshold:
        print('INFO: Using pruning threshold', threshold)
    print('INFO:', len(data_S), 'testing sentences')
    for sentence in data_S:
        if modified:
            prediction = viterbi_prunned_modified(sentence, tagsetT, pwt, ptt, possible_next_tags, unknown_states=unk,
                                                  tags_dict=possible_tags, threshold=threshold)
        else:
            if threshold:
                prediction = viterbi_prunned(sentence, tagsetT, pwt, ptt, possible_next_tags, unknown_states=unk,
                                             tags_dict=possible_tags, threshold=threshold)
            else:
                prediction = vite(sentence, tagsetT, pwt, ptt, possible_next_tags, unknown_states=unk,
                                  tags_dict=possible_tags)
        # at the beginning there should be two ###
        # at the end there should be two ~~~
        if prediction[0] != '###' and prediction[1] != '###':
            print('ERROR: Something went horribly wrong')
            continue
        if prediction[-1] != '###' and prediction[-2] != '###':
            print('ERROR: Something went horribly wrong')
            continue
        if len(prediction[-1]) != len(prediction[-2]):
            print('ERROR: Something went horribly wrong')
            continue
        for i in range(2, len(prediction) - 2):
            total += 1
            if prediction[i] == sentence[i][1]:
                correct += 1
    print('INFO: accuracy without starting and ending tags:', correct / total)
