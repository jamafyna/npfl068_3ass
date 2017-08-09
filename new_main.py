#!/usr/bin/env python3
from collections import defaultdict
from optparse import OptionParser
from os.path import basename

from functions import get_initial_parameters, fix_sentence_boundaries, evaluate_test_data, fix_sentence_boundaries_words
from functions2 import baum_welch
from classes import Pwt, Ptt

parser = OptionParser(usage="usage: %prog [options] filename count")

parser.add_option("-f", "--fold", type="int", dest="fold", default=0,
                  help="Specify the number of fold")

parser.add_option("-u", "--unknown-states",
                  action="store_true", dest="unknown", default=False,
                  help="Use only the states from the training data, enforces smoothing")

parser.add_option("-o", "--output",
                  type="string", dest="outputdir", default="data",
                  help="Use only the states from the training data, enforces smoothing")

(options, args) = parser.parse_args()
fold = options.fold
unk = options.unknown
threshold = 20
file_name = args[0]  # 'data/texten2.ptg'
if unk:
    dest = options.outputdir + '/' + basename(file_name) + 'full'
else:
    dest = options.outputdir + '/' + basename(file_name)
print('INFO: Processing the file "', file_name, '"')

file = open(file_name, encoding="iso-8859-2", mode='rt')
data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))

dataE = data[:10000]  # data for estimating the raw probabilities
# strip off the tags from the remaining data
dataT = [w for w, _ in data[10000:-60000]]  # training data
data_T = fix_sentence_boundaries_words(dataT)
print('INFO:', len(data_T), 'training sentences')

dataH = data[-60000:-40000]  # held_out data
dataS = data[-40000:]  # testing data
data_S = fix_sentence_boundaries(dataS)
print('INFO:', len(data_S), 'testing sentences')
# double separators in estimation data set
new = []
for e in dataE:
    if e == ('###', '###'):
        new.append(e)
        new.append(e)
    else:
        new.append(e)
dataE = new
new = []
# double separators in held out data
for e in dataH:
    if e == ('###', '###'):
        new.append(e)
        new.append(e)
    else:
        new.append(e)
dataH = new

# estimate the raw parameters and smooth them
p, trig_tag_set = get_initial_parameters([t for w, t in dataE])
guess_tags = [t for (_, t) in dataE]
# extract the states from the data --- only the seen bigrams
state_set = set(zip(guess_tags, guess_tags[1:]))

tagsetT = set([t for (_, t) in dataE])  # set of tags
wordsetT = set([w for (w, _) in dataE])  # set of words
# expand states to not ban any beginning
# for tag in tagsetT:
#     state_set.add((tag, '###'))
#     state_set.add(('###', tag))
# organize them into dictionaries
if unk:
    possible_next = defaultdict(lambda: tagsetT)
    possible_prev = defaultdict(lambda: tagsetT)
else:
    possible_next = defaultdict(lambda: set())
    possible_prev = defaultdict(lambda: set())
    for u, v in state_set:
        possible_next[u].add(v)
        possible_prev[v].add(u)

pwt = Pwt(dataE, len(wordsetT), len(tagsetT))
ptt = Ptt(p, [t for (_, t) in dataH], None)
my_pwt_distrib = defaultdict(lambda: 0)

# touch all the training data so that the probabilities are precomputed
for word in set(dataT):
    for s in state_set:
        my_pwt_distrib[word, s] = pwt.p(word, s[1])

for u, v in state_set:
    for w in possible_next[v]:
        ptt.p(u, v, w)
pwt.distribution['###', '###'] = 1

# guesstimate the accuracy before the training
# evaluate_test_data(data_S, tagsetT, pwt, ptt, possible_next, threshold=threshold, unk=unk)

# train the model
e_p, t_p = baum_welch(ptt.distribution, my_pwt_distrib, data_T, possible_next, possible_prev, state_set, file=dest,
                      fold=fold)

# smooth the distributions


# estimate the accuracy after training
# evaluate_test_data(data_S, tagsetT, pwt, ptt, possible_next, threshold=threshold, unk=unk)
