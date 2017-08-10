import warnings
from collections import Counter, defaultdict

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
    p_t2 = Counter(bigram_tags)
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
    return [p_t0, p_t1, p_t2, p_t3], set(trigram_tags)


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
        tagged.append(last[0])
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
                    q = alpha_t[u, v] * transition_p.p(u, v, w) * emission_p.p(sentence[time][0], (v, w))
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
        tagged.append(last[0])
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
            # the default value of threshold is None, which leads to no pruning at all
            prediction = viterbi_prunned(sentence, tagsetT, pwt, ptt, possible_next_tags, unknown_states=unk,
                                             tags_dict=possible_tags, threshold=threshold)
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
