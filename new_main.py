#!/usr/bin/env python3
from collections import defaultdict
from optparse import OptionParser
from os.path import basename
import dill

from functions import get_initial_parameters, fix_sentence_boundaries, evaluate_test_data, fix_sentence_boundaries_words
from functions2 import baum_welch
from classes import Pwt, Ptt, PttModified, PwtModified

parser = OptionParser(usage="usage: %prog [options] filename [transition probability file] [emission probability file]")

parser.add_option("-f", "--fold", type="int", dest="fold", default=0,
                  help="Specify the number of fold")

parser.add_option("-t", "--threshold", type="int", dest="threshold", default=10,
                  help="Specify the threshold for evaluation pruning")

parser.add_option("-u", "--unknown-states",
                  action="store_true", dest="unknown", default=False,
                  help="Use only the states from the training data, enforces smoothing")

parser.add_option("-o", "--output",
                  type="string", dest="outputdir", default=".",
                  help="Use only the states from the training data, enforces smoothing")

parser.add_option("-e", "--evaluate",
                  action="store_true", dest="evaluate", default=False,
                  help="Option used for evaluating the data")

(options, args) = parser.parse_args()
fold = options.fold
unk = options.unknown
evaluate = options.evaluate
threshold = 20
file_name = args[0]
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

dataE = dataT[:10000]  # data for estimating the raw probabilities
# guesstimate the size of the vocabulary
vocab_size = len(set([w for w, t in dataT]))
# strip off the tags from the remaining data
dataT = [w for w, _ in dataT[10000:]]  # training data
data_T = fix_sentence_boundaries_words(dataT)
print('INFO:', len(data_T), 'training sentences')

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
# find states
tagsetT = set([t for (_, t) in dataE])  # set of tags
wordsetT = set([w for (w, _) in dataE])  # set of words
if unk:
    state_set = [(x, y) for x in tagsetT for y in tagsetT]
else:
    # extract the states from the data --- only the seen bigrams
    guess_tags = [t for (_, t) in dataE]
    state_set = set(zip(guess_tags, guess_tags[1:]))

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
ptt = Ptt(p, [t for (_, t) in dataH])
my_pwt_distrib = defaultdict(lambda: 0)

# touch all the training data so that the probabilities are precomputed
for word in set(dataT):
    for s in state_set:
        my_pwt_distrib[word, s] = pwt.p(word, s[1])

for u, v in state_set:
    for w in possible_next[v]:
        ptt.p(u, v, w)
pwt.distribution['###', '###'] = 1

if evaluate:
    # guesstimate the accuracy before the training
    evaluate_test_data(data_S, tagsetT, pwt, ptt, possible_next, threshold=10, unk=unk)
    # load the distributions from the dill files
    ptt = dill.load(open(args[1], 'rb'))
    trans = PttModified(ptt, state_set, possible_next)
    pwt = dill.load(open(args[2], 'rb'))
    state_counts = defaultdict(lambda: 0)
    for s in list(zip(guess_tags, guess_tags[1:])):
        state_counts[s] += 1
    emit = PwtModified(pwt, state_counts, vocab_size)
    evaluate_test_data(data_S, tagsetT, emit, trans, possible_next, threshold=10, modified=True, unk=unk)
    # smooth the distributions
    print('Evaluate')

else:  # train the model
    e_p, t_p = baum_welch(ptt.distribution, my_pwt_distrib, data_T, possible_next, possible_prev, state_set, file=dest,
                          fold=fold)
