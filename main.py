#!/usr/bin/env python3
from collections import defaultdict
from optparse import OptionParser

from functions import baum_welch, fix_sentence_boundaries, vite, get_initial_parameters, viterbi_prunned
from classes import LinearSmoothedDistribution, Ptt
from classes import PwtUnknown, Pwt, PwtUnknownSmooth

parser = OptionParser(usage="usage: %prog [options] filename count")

parser.add_option("-s", "--supervised",
                  action="store_true", dest="supervised", default=True,
                  help="Use the supervised method (the default option)")

parser.add_option("-u", "--unsupervised",
                  action="store_false", dest="supervised", default=False,
                  help="Use the unsupervised method (Baum-Welch)")

parser.add_option("-o", "--out-of-vocabulary",
                  action="store_true", dest="oov", default=False,
                  help="Replace low frequency words with OOV token")

parser.add_option("-l", "--smooth-lexical",
                  action="store_true", dest="lex", default=False,
                  help="Smooth the lexical model by adding less than one")

parser.add_option("-k", "--known-states",
                  action="store_true", dest="known", default=False,
                  help="Use only the states from the training data, enforces smoothing")

parser.add_option("-n", "--threshold", type="int", dest="num", default=0,
                  help="Specify the number of best trellis states used fo generating the next stage (default is 10)")

parser.add_option("-f", "--fold", type="int", dest="fold", default=0,
                  help="Specify the number of fold")

(options, args) = parser.parse_args()
file_name = args[0]
print('INFO: Processing the file "', file_name, '"')
unk = not options.known
if not unk:
    print('INFO: Limiting the sates of states to the states from the training data')
lex = options.lex
oov = options.oov
supervised = options.supervised
threshold = options.num
fold = options.fold

file = open(file_name, encoding="iso-8859-2", mode='rt')
data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))

# split the data
if fold == 0:
    dataT = data[:-60000]  # training data
    dataH = data[-60000:-40000]  # held_out data
    dataS = data[-40000:]  # testing data
elif fold == 1:
    dataT = data[60000:]  # training data
    dataH = data[40000:60000]  # held_out data
    dataS = data[:40000]  # testing data
elif fold == 2:
    dataT = data[:60000] + data[120000:]  # training data
    dataH = data[60000:80000]  # held_out data
    dataS = data[80000:120000]  # testing data
elif fold == 3:
    dataT = data[:120000] + data[180000:]  # training data
    dataH = data[160000:180000]  # held_out data
    dataS = data[120000:160000]  # testing data
else:
    dataT = data[:40000] + data[100000:]  # training data
    dataH = data[80000:100000]  # held_out data
    dataS = data[40000:80000]  # testing data

data_S = fix_sentence_boundaries(dataS)

STARTw = '###'
STARTt = '###'
new = []
for e in dataT:
    if e == ('###', '###'):
        new.append(e)
        new.append(e)
    else:
        new.append(e)
dataT = new
new = []
for e in dataH:
    if e == ('###', '###'):
        new.append(e)
        new.append(e)
    else:
        new.append(e)
dataH = new

# dataT = [('###', '###'), ('###', '###'), ('A', 'A'), ('B', 'B'), ('A', 'A'), ('C', 'C'), ('~~~', '~~~'), ('~~~', '~~~')]
# dataH = [('###', '###'), ('###', '###'), ('A', 'A'), ('B', 'B'), ('~~~', '~~~'), ('~~~', '~~~')]
# dataS = [('###', '###'), ('###', '###'), ('A', 'A'), ('C', 'C'), ('~~~', '~~~'), ('~~~', '~~~')]

# flat_data = [t for s in data_T for (_, t) in s]
p, trig_tag_set = get_initial_parameters([t for w, t in dataT])
tagsetT = set([t for (_, t) in dataT])  # set of tages
wordsetT = set([w for (w, _) in dataT])  # set of words

possible_tags = None
if oov:  # estimate unknown words by rare
    print('INFO: Estimating distribution of unknown words from the rare words')
    if lex:  # smooth
        print('INFO: Using smoothed lexical distribution')
        pwt = PwtUnknownSmooth(dataT, len(wordsetT), len(tagsetT))
    else:  # do not smoot
        print('INFO: Using non-smoothed distribution')
        pwt = PwtUnknown(dataT, len(wordsetT), len(tagsetT))
        # because the distribution is not smoothed, we can speed things up by remembering tags
        # that yield non zero probability, i. e. the observed word/tag pairs
        possible_tags_tmp = defaultdict(lambda: set())
        for w, t in pwt.wt_counts.keys():
            possible_tags_tmp[w].add(t)
        default_set = possible_tags_tmp['@UNK']
        possible_tags = defaultdict(lambda: default_set)
        for w, t in pwt.wt_counts.keys():
            possible_tags[w] = possible_tags_tmp[w]
        del possible_tags_tmp



else:
    print('INFO: All unknown words are the same')
    pwt = Pwt(dataT, len(wordsetT), len(tagsetT))
# ptt = LinearSmoothedDistribution(dataH, p[0], p[1], p[2], p[3])
ptt = Ptt(p, [t for (_, t) in dataH], [t for (_, t) in dataT])
possible_next_tags = defaultdict(lambda: set())
for x, y in zip([t for (_, t) in dataT], [t for (_, t) in dataT][1:]):
    possible_next_tags[x].add(y)
# for key in possible_next_tags.keys():
#     possible_next_tags[x].add('~~~')

# sentencesS = fix_sentence_boundaries(dataS)

total = 0
correct = 0
if threshold:
    print('INFO: Using prunning threshold', threshold)
print('INFO:', len(data_S), 'testing sentences')
for sentence in data_S:
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
            # print(correct, total, correct / total)
print('INFO: accuracy without starting and ending tags:', correct / total)



# # Baum-Welch training
# transition_matrix, emission_matrix = baum_welch(dataT, dataH)
# # save the trained model
# np.save('transition_matrix.npy', transition_matrix)
# np.save('emission_matrix.npy', emission_matrix)



# Mean accuracy of trivial tagger: 0.865374314735
# Standard deviation: 0.0124922171647
# vladan@leviatan:~/PycharmProjects/npfl068$ ./assignment3_1.py -r 10 data/textcz2.ptg
# Mean accuracy of trivial tagger: 0.738888267001
# Standard deviation: 0.0134321134672
