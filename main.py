#!/usr/bin/env python3
from collections import defaultdict
from functions import baum_welch, fix_sentence_boundaries, vite, get_initial_parameters
from classes import LinearSmoothedDistribution
from classes import PwtUnknown as Pwt

import numpy as np
import sys

from nltk import UnigramTagger

file_name = 'data/texten2.ptg'  # sys.argv[1]  #
file = open(file_name, encoding="iso-8859-2", mode='rt')

data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))

dataT = data[:-40000]  # training data
data_T = fix_sentence_boundaries(dataT)
dataH = data[-60000:-40000]  # held_out data
data_H = fix_sentence_boundaries(dataH)
dataS = data[-40000:]  # testing data
data_S = fix_sentence_boundaries(dataS)
print('INFO:', len(data_S), 'testing sentences')

# dataT = [('###', '###'), ('###', '###'), ('A', 'A'), ('B', 'B'), ('A', 'A'), ('C', 'C'), ('~~~', '~~~'), ('~~~', '~~~')]
# dataH = [('###', '###'), ('###', '###'), ('A', 'A'), ('B', 'B'), ('~~~', '~~~'), ('~~~', '~~~')]
# dataS = [('###', '###'), ('###', '###'), ('A', 'A'), ('C', 'C'), ('~~~', '~~~'), ('~~~', '~~~')]
flat_data = [t for s in data_T for (_, t) in s]
p, trig_tag_set = get_initial_parameters(flat_data)
tagsetT = set([t for (_, t) in dataT])  # set of tages
tagsetT.add('~~~')
wordsetT = set([w for (w, _) in dataT])  # set of words
wordsetT.add('~~~')
pwt = Pwt(dataT, len(wordsetT), len(tagsetT))
ptt = LinearSmoothedDistribution(dataH, p[0], p[1], p[2], p[3])
possible_next_tags = defaultdict(lambda: set())
for x, y in zip(flat_data, flat_data[1:]):
    possible_next_tags[x].add(y)
for key in possible_next_tags.keys():
    possible_next_tags[x].add('~~~')

# sentencesS = fix_sentence_boundaries(dataS)

total = 0
correct = 0
for sentence in data_S:
    prediction = vite(sentence, tagsetT, pwt, ptt, possible_next_tags)
    # at the beginning there should be two ###
    # at the end there should be two ~~~
    if prediction[0] != '###' or prediction[1] != '###':
        print('ERROR: Something went horribly wrong')
        continue
    if prediction[-1] != '~~~' or prediction[-2] != '~~~':
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
