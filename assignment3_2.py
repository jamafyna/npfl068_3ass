#!/usr/bin/env python3
import sys
import math
import random
import datetime
import warnings
import numpy as np
from optparse import OptionParser
from sys import stdin as sin
from sys import stdout as sout
from collections import Counter
from classes import LinearSmoothedDistribution, Pwt, Ptt
from collections import Counter
from functions import get_initial_parameters, viterbi, baum_welch

STARTw = "###"  # start token/token which split sentences
STARTt = "###"  # STARTw for words, STARTt for tags

# ---------------------PUVODNI VITERBI ----------------------------------------

def viterbi_fast(text, tagset, wordset, trigramtagset, Pwt, Ptt, start, usetrigram):
    """
    Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset),
    and first half of a start state (usually end of previous sentence or start token).
    """
    if len(text) == 0: return []
    isOOV = False  # indicates if proceeded word is out-of-vocabulary
    if text[0] == STARTw:
        warnings.warn("inconsistend data, start tokens", Warning)
    else:
        text = [(STARTw, STARTt)] + text
    V = {}  # structure for remember viterbi computation
    # V[time][state]=(probability,path) where state==(tag1,tag2)
    path = {}  # the best path to some state
    V[0] = {}
    V[1] = {}
    OOVcount = 0
    # --- initialisation, starting state
    V[0][start] = (1, [start])
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
        if w not in wordset:
            OOVcount += 1
            isOOV = True
        for t in tagset:
            if t == STARTt: continue
            bests = {}
            bestpath = []
            maxprob = 0
            for (i, j) in V[prev]:
                if (usetrigram and ((i, j, t) not in trigramtagset)): continue
                value = V[prev][i, j][0] * get_ptt_f(i, j, t)
                if value >= maxprob:  # '=' because of very small numbers
                    bests[0] = i
                    bests[1] = j
                    maxprob = value
                    bestpath = V[prev][i, j][1]
            if bests != {}:
                V[now][bests[1], t] = (maxprob * (Pwt.get_smoothed_pwt(w, t, isOOV)), bestpath + [(w, t)])
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


def viterbi2(text, tagset, wordset, trigramtagset, Pwt, Ptt, start, usetrigram=True):
    """
    Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset),
    and first half of a start state (usually end of previous sentence or start token).
    """
    if not usetrigram: warnings.warn("viterbi2 always uses only known trigrams")
    if len(text) == 0: return []
    isOOV = False  # indicates if proceeded word is out-of-vocabulary
    if text[0] == STARTw:
        warnings.warn("inconsistend data, start tokens", Warning)
    else:
        text = [(STARTw, STARTt)] + text
    V = {}  # structure for remember viterbi computation
    # V[time][state]=(probability,path) where state==(tag1,tag2)
    V[0] = {}
    V[1] = {}
    OOVcount = 0
    # --- initialisation, starting state
    V[0][start] = (0, [start])
    V[1][STARTt, STARTt] = (1, [start, (STARTw, STARTt)])
    now = 1  # value depends on k which starts from 1
    prev = 0

    # --- finding the best way
    for k in range(2, len(text) + 1):
        isOOV = False
        w = text[k - 1]
        now = k % 2  # k modulo 2 instead of k; it is sufficient to remember
        prev = (now + 1) % 2  # instead of k-1;          only actual and previous time
        V[now] = {}
        if w not in wordset:
            OOVcount += 1
            isOOV = True
        for (i, j, t) in trigramtagset:
            if (i, j) not in V[prev]:
                continue
            value = V[prev][i, j][0] * Ptt.get_ptt(i, j, t)
            if ((j, t) not in V[now]) or value > V[now][j, t][0]:  # '=' because of very small numbers
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


# ----------------- VITERBI ----------------------------------------------




# ------------------- EVALUATION ------------------------------------------

def accuracy(right, computed):
    if len(right) != len(computed):
        warnings.warn("inconsistend data, different length of test and computed solution", Warning)
    correct = 0
    allc = min(len(right), len(computed))
    for i in range(0, allc - 1):
        if right[i][0] != computed[i][0]:
            raise Exception("inconsistent data, different words in test and computed")
        if right[i][1] == computed[i][1]:
            correct += 1
    print("right tagged: ", correct, ", all: ", allc)
    return correct / allc


############################################################################
################# MAIN PROGRAM STARTS ######################################
############################################################################


# -----------------------------initialization-------------------------------

# ----- parsing arguments ---------

parser = OptionParser(usage="usage: %prog [options] filename count")
parser.add_option("-s", "--supervised",
                  action="store_true", dest="supervised", default=False,
                  help="Use supervised method (the default is unsupervised)")
parser.add_option("-m", "--memory",
                  action="store_true", dest="memory", default=False,
                  help="Use supervised method (the default is unsupervised)")
parser.add_option("-u", "--unsupervised",
                  action="store_false", dest="supervised", default=False,
                  help="Use unsupervised method (the default option)")
parser.add_option("-v", "--viterbi2",
                  action="store_true", dest="viterbi", default=False,
                  help="Use the second variant of viterbi")
parser.add_option("-f", "--fast",
                  action="store_true", dest="fast", default=False,
                  help="Use a faster algorithmus similar to Viterbi")
parser.add_option("-t", "--unknowntrigrams",
                  action="store_false", dest="trigrams", default=True,
                  help="Use a faster algorithmus similar to Viterbi")

(options, args) = parser.parse_args()
file_name = 'data/texten2.ptg'  # args[0]
file = open(file_name, encoding="iso-8859-2", mode='rt')
supervised = options.supervised
memory = options.memory
viterbi_bool = options.viterbi
fast = options.fast
usetrigram = options.trigrams

# ------ data preparation ---------

data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))
dataT = data[:-60000]  # training data
dataH = data[-60000:-40000]  # held_out data
dataS = data[-40000:]  # testing data
# dataS = data[-400:]  # testing data
if dataS[0] != [(STARTw, STARTt)]:
    dataS = [(STARTw, STARTt)] + dataS
data = []  # for gc
OOVcount = 0  # unknown words
tagsetT = set([t for (_, t) in dataT])  # set of tages
wordsetT = set([w for (w, _) in dataT])  # set of words

# ------- computation -------------

if supervised:
    pp, trig_tag_set = get_initial_parameters([t for (_, t) in dataT])
    # get distribution of tags from train data, not smoothed yet
else:
    # Baum-Welch training
    transition_matrix, emission_matrix = baum_welch(dataT, dataH)
    # save the trained model
    np.save('transition_matrix.npy', transition_matrix)
    np.save('emission_matrix.npy', emission_matrix)
    sys.exit()

# sem by šlo dát i dataH, resp. cele p_wt spocitat i z heldout - zabiralo hodne pameti
pwt = Pwt(dataT, len(wordsetT), len(tagsetT))
# distribution of pairs (word,tag),smoothed

pt = Ptt(pp, [t for (_, t) in dataH], [t for (_, t) in dataT], memory)
ptt = LinearSmoothedDistribution(dataH, pp[0], pp[1], pp[2], pp[3])
# pt = LinearSmoothedDistribution(dataH, pp[0], pp[1], pp[2], pp[3])
# distrib. of tags, smoothed

# -------- tagging by Viterbi----------------
tagged = []
sentence = []
v = []
c = 0
sentence_end = (STARTw, STARTt)
# run Viterbi on each sentence of the test data
print("--- Starting Viterbi --- ")

# choosing which algorithm we want to use
if viterbi_bool:
    viterbialg = viterbi2
elif fast:
    viterbialg = viterbi_fast
else:
    viterbialg = viterbi

exit()
for p in dataS:
    if p == (STARTw, STARTt):
        if (sentence != []):
            v, c = viterbialg([w for (w, _) in sentence], tagsetT, wordsetT, trig_tag_set, pwt, pt, sentence_end,
                              usetrigram)
        tagged = tagged + v + [(STARTt, STARTw)]
        # for t in tagged: print(t)
        OOVcount += c
        if len(sentence) == 0:
            sentence_end = (STARTw, STARTt)
        else:
            sentence_end = sentence[len(sentence) - 1]
        sentence = []
        v = []
        c = 0
    else:
        sentence = sentence + [p]

print("---VITERBI ENDED ---")
print('out-of-vocabulary words:', OOVcount)
o = accuracy(dataS, tagged)
print('accuracy:', o)
