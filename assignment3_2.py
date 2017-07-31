#!/usr/bin/env python3
import sys
import math
import random
import datetime
import warnings
import itertools
from optparse import OptionParser
from sys import stdin as sin
from sys import stdout as sout
from collections import Counter
from classes import LinearSmoothedDistribution
from classes import AddOneSmoothedDistribution


STARTw = "###"
STARTt = "###"


def get_prob(p, h):
    """ Returns a corresponding value from the given tuple if an item is in the tuple, otherwise returns zero."""
    if h in p:
        return p[h]
    else:
        return 0


# ----- estimating parametres, supervised learning --------------

def get_parametres_superv(tags):
    """
    Computes uniform, unigram, bigram and trigram distributions from given data.
    """
    tags_uniq = Counter(tags)
    # constant probability mass (avoiding zeros)
    p_t0 = 1 / len(tags_uniq)
    # unigram probabilities
    p_t1 = {t: tags_uniq[t] / len(tags) for t in tags_uniq}
    # bigram probabilities
    bigram_tags = Counter(zip(tags[:-1], tags[1:]))
    p_t2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}
    # trigram probabilities
    trigram_tags = Counter([trig for trig in zip(tags[:-2], tags[1:-1], tags[2:])])
    p_t3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}
    p_tt = [p_t0, p_t1, p_t2, p_t3]
    return p_tt


# ----------------- smoothing EM algorithm------------------------------------------

def EMiter(data, p, l):
    """An one iteration of EM algorithm."""
    tri = [u for u in zip(data[:-2], data[1:-1], data[2:])]
    pp = {
        (i, j, k): l[3] * get_prob(p[3], (i, j, k)) + l[2] * get_prob(p[2], (j, k)) + l[1] * get_prob(p[1], k) + l[0] * p[0] for (i, j, k) in set(tri)}  # new p'(lambda)
    c = [0, 0, 0, 0]
    for (i, j, k) in tri:
        pptemp = pp[(i, j, k)]
        c[0] = c[0] + l[0] * p[0] / pptemp
        c[1] += l[1] * get_prob(p[1], k) / pptemp
        c[2] += l[2] * get_prob(p[2], (j, k)) / pptemp
        c[3] += l[3] * get_prob(p[3], (i, j, k)) / pptemp
    return [i / sum(c) for i in c]  # normalised


def EMalgorithm(data, p):  # heldoutdata
    """EM algorithm, input: data: heldout data, p: probabilities counted form training data """
    l = [10, 10, 10, 10]  # infinity, due to first while
    nextl = [0.25, 0.25, 0.25, 0.25]  # lambdas counted in the next iteration
    e = 0.001  # precision
    itercount = 0
    while (abs(l[0] - nextl[0]) >= e or abs(l[1] - nextl[1]) >= e or abs(l[2] - nextl[2]) >= e or abs(
                l[3] - nextl[3]) >= e):  # expected precision is not yet achieved
        l = nextl
        nextl = EMiter(data, p, l)
        itercount = itercount + 1
    sout.write("\nSmoothing, EM algorithm:\n")
    sout.write("\nnumber of iterations:" + str(itercount) + ", precision: " + str(e) + "\n")
    return nextl


# ----------------------------------------------------------------------------------
def smoothEM(p, heldout, traindata):
    """
    Do linear interpolation trigram smoothing, need estimated probabilities form train data, heldout data fo estimate lambdas and testing data.
    """
   
    l = EMalgorithm(heldout, p)  # get lambdas
    print("lambdas:",l)
    tri = [i for i in zip(traindata[:-2], traindata[1:-1], traindata[2:])]
    ttrainset=set(traindata) 
    pt_em = {
        (i, j, k): (
            l[0] * p[0] + l[1] * get_prob(p[1], k) + l[2] * get_prob(p[2], (j, k)) + l[3] * get_prob(p[3], (i, j, k)))
        # for (i, j, k) in tri}
        for i in ttrainset for j in ttrainset for k in ttrainset}
    return pt_em


def smoothAdd(pc, data, tagset, lamb=2**(-10)):
    """
    Do smoothing by Adding less than 1, need counts of c(t,w) and c(h) from train data and need test data for smooting.
    """
    pwt = {(w, t): ((get_prob(pc[0], (w, t)) + lamb) / (get_prob(pc[1], t) + lamb*len(wordsetT) * len(tagset))) for w in wordsetT for t in tagset}
    return pwt


class Pwt:
    wt_counts = []
    t_counts = []
    vocab_size=0

    def __init__(self,data,vocabulary_size):
        self.t_counts = Counter([t for (_,t) in data])
        self.wt_counts = Counter(data)  # Counter([(w, t) for (w, t) in data_wt])
        self.vocab_size=vocabulary_size

    def get_smoothed_pwt(self, w, t, isOOV):
        """
        Returns smoothed p(w|t), by smoothing less than 1. Suppose that w and t are from known wordset and tagset, not unknown.
        """
        return (self.wt_counts[w,t]+1)/(self.t_counts[t]+self.vocab_size)

    #def get_pwt(self, w, t, isOOV=False, lamb=2**(-10)):
    #    if isOOV: return 1/self.len_tagset # if the w is out-of-vocabulary, then use uniform distribution
    #    return ((get_prob(self.wt_counts, (w, t)) + lamb) / (
    #       get_prob(self.t_counts, t) + lamb * self.len_wordset * self.len_tagset))

class Ptt:
    """
    Class for getting smoothed arc probability
    """
    # TODO: ČASEM možná vylepšit na nepamatování si celé tabulky, ale dynam.počítání
    p_t = []

    def __init__(self, p, heldout, train):
        print(len(heldout), len(train))
        self.p_t = smoothEM(p, heldout, train)  # probabilities p_t1,p_t2,p_t3

    def get_ptt(self, t1, t2, t3):
        """
        Returns smoothed p(t3|t1,t2).
        """
        return self.p_t[t1, t2, t3]  # TODO: Možná udělat časem dynamicky


def viterbilog(text,tagset,wordset,Pwt,Ptt):
        if len(text)==0: return []
        isOOV=False # indicates if proceeded word is out-of-vocabulary
        tagsetcontainsSTART=False
        while STARTt in tagset: 
            tagset.remove(STARTt)
            tagsetcontainsSTART=True 
            # we dont want to have this tag in the middle of the tag sequence,  
            # for sure while instead of only remove
        if text[0]==STARTw: 
            warnings.warn("inconsistend data, start tokens",Warning)
        else: text=[(STARTw, STARTt), (STARTw, STARTt)]+text 
        V={}    # structure for remember viterbi computation
                # V[time][state]=(probability,path) where state==(tag1,tag2)
        path={} # the best path to some state
        V[0]={} 
        V[1]={}
        # --- initialisation, starting state
        V[1][STARTt,STARTt]=(0,[(STARTw,STARTt),(STARTw,STARTt)])
        OOVcount=0
        # --- finding the best way
        for k in range(2,len(text)):
                isOOV=False
                w=text[k]
                now=k%2        # k modulo 2 instead of k; it is sufficient to remember 
                prev=(now+1)%2 # instead of k-1;          only actual and previous time
                V[now]={}
                if w not in wordset:
                    OOVcount+=1
                    isOOV=True
                for t in tagset:
                    bests={}
                    bestpath=[]
                    maxprob=-float('inf')
                    for (i,j) in V[prev]:
                        value=V[prev][i,j][0]+math.log(Ptt.get_ptt(i,j,t),2)
                        if value>=maxprob: # '=' because of very small numbers  
                            bests[0]=i
                            bests[1]=j
                            maxprob=value
                            bestpath=V[prev][i,j][1]
                    V[now][bests[1],t]=(maxprob+math.log(Pwt.get_smoothed_pwt(w,t,isOOV),2),bestpath+[(w,t)])
        if tagsetcontainsSTART: tagset.add(STARTt)  # to be the same as at start
        # --- final search the best tag sequence
        maxprob=-float('inf')
        ends={}              # the best end state
        for s in V[now]:
                if V[now][s][0]>=maxprob:
                    maxprob=V[now][s][0]
                    ends=s
        if(maxprob==-float('inf')):warnings.warn("not changed max proability at the end")
        return V[now][ends][1][2:],OOVcount # only [2:] because of start tokens

def viterbi(text,tagset,wordset,Pwt,Ptt,start):
        """
        Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset), and first half of a start state (usually end of previous sentence or start token).
        """
        if len(text)==0: return []
        isOOV=False # indicates if proceeded word is out-of-vocabulary
        tagsetcontainsSTART=False
        while STARTt in tagset: 
            tagset.remove(STARTt)
            tagsetcontainsSTART=True 
            # we dont want to have this tag in the middle of the tag sequence,  
            # for sure while instead of only remove
        if text[0]==STARTw: 
            warnings.warn("inconsistend data, start tokens",Warning)
        else: text=[(STARTw, STARTt)]+text 
        V={}    # structure for remember viterbi computation
                # V[time][state]=(probability,path) where state==(tag1,tag2)
        path={} # the best path to some state
        V[0]={} 
        V[1]={}
        OOVcount=0
        # --- initialisation, starting state
        #V[1][STARTt,STARTt]=(1,[(STARTw,STARTt),(STARTw,STARTt)])
        V[0][start]=(0,[start])
        V[1][STARTt,STARTt]=(1,[start,(STARTw,STARTt)])
        now=1 # value depends on k which starts from 1
        prev=0
        # --- finding the best way
        for k in range(2,len(text)+1):
                isOOV=False
                w=text[k-1]
                now=k%2        # k modulo 2 instead of k; it is sufficient to remember 
                prev=(now+1)%2 # instead of k-1;          only actual and previous time
                V[now]={}
                if w not in wordset:
                    OOVcount+=1
                    isOOV=True
                for t in tagset:
                    bests={}
                    bestpath=[]
                    maxprob=0
                    for (i,j) in V[prev]:
                        value=V[prev][i,j][0]*Ptt.get_ptt(i,j,t)
                       # if value==0: Chtělo by to aspoň takovýto treshold, zahodit všechny stavy mající value=0, což může být velmi velmi malé číslo zaokrouhlené na 0
                        if value>=maxprob: # '=' because of very small numbers  
                            bests[0]=i
                            bests[1]=j
                            maxprob=value
                            bestpath=V[prev][i,j][1]
                    V[now][bests[1],t]=(maxprob*Pwt.get_smoothed_pwt(w,t,isOOV),bestpath+[(w,t)])
        if tagsetcontainsSTART: tagset.add(STARTt)  # to be the same as at start
        # --- final search the best tag sequence
        maxprob=0
        ends={}              # the best end state
        for s in V[now]:
                if V[now][s][0]>=maxprob:
                    maxprob=V[now][s][0]
                    ends=s
        if maxprob==0 & len(text)>1: 
            warnings.warn("zero maxprobability at the end")
        return V[now][ends][1][2:],OOVcount # only [2:] because of start tokens


def occuracy(right,computed):
       if len(right)!=len(computed): 
            warnings.warn("inconsistend data, different length of test and computed solution",Warning)
       correct=0
       allc=min(len(right),len(computed))
       for i in range(0,allc-1):
           if right[i][0]!=computed[i][0]:
               raise Exception("inconsistent data, different words in test and computed")
           if right[i][1]==computed[i][1]: correct+=1
       return (correct,allc)

# -----------------------------initialization-------------------------------

# ----- parsing arguments ---------
parser = OptionParser(usage="usage: %prog [options] filename count")
parser.add_option("-s", "--supervised",
                  action="store_true", dest="supervised", default=False,
                  help="Use supervised method (the default is unsupervised)")
parser.add_option("-u", "--unsupervised",
                  action="store_false", dest="supervised", default=False,
                  help="Use unsupervised method (the default option)")
(options, args) = parser.parse_args()
file_name = args[0]
file = open(file_name, encoding="iso-8859-2", mode='rt')
supervised = options.supervised

# ------ data preparation ---------

data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))
dataT = [(STARTw, STARTt), (STARTw, STARTt)] + data[:-60000]  # training data
dataH = [(STARTw, STARTt), (STARTw, STARTt)] + data[-60000:-40000]  # held_out data used for smoothing
dataS = [(STARTw, STARTt), (STARTw, STARTt)] + data[-40000:]  # testing data
#dataS = data[-39:] # testingdata for debuging
if dataS[0]!=[(STARTw,STARTt)]: dataS= [(STARTw,STARTt)]+dataS
data = []  # for gc
OOVcount=0
tagsetT = set([t for (_, t) in dataT])
wordsetT = set([w for (w, _) in dataT])

# ------- computation -------------

if supervised:
    pp = get_parametres_superv(dataT)  # get p_t from train data, not smoothed yet
else:
    print("todo")
    sys.exit()
#p_t = smoothEM(pp[1], [t for (_, t) in dataH], [t for (_, t) in dataT])  # probabilities p_t1,p_t2,p_t3
#p_wt=smoothAdd(pp[2],[w for (w,_) in dataT],set([t for (_,t) in dataT]))
#sem by šlo dát i dataH, resp. cele p_wt spocitat i z heldout - zabiralo hodne pameti

pwt = Pwt(dataT,len(tagsetT)*len(wordsetT))
pt = Ptt(pp, [t for (_, t) in dataH], [t for (_, t) in dataT])

#d = LinearSmoothedDistribution(dataH, pp[0],pp[1], pp[2] ,pp[3])
#h = AddOneSmoothedDistribution(dataH,dataT)
# -------- tagging ----------------
#sys.exit()
tagged=[]
sentence=[]
v=[]
c=0
sentence_end=(STARTw,STARTt)
for p in dataS:
    if p==(STARTw,STARTt): 
        if(sentence!=[]):
            v,c=viterbi([w for (w,_) in sentence], tagsetT, wordsetT, pwt, pt, sentence_end)
            v,c=viterbilog([w for (w,_) in sentence], tagsetT, wordsetT, pwt, pt)
        tagged=tagged+v+[(STARTt,STARTw)]
        OOVcount+=c
        if len(sentence)==0: sentence_end=(STARTw,STARTt)
        else: sentence_end=sentence[len(sentence)-1]
        sentence=[]
        v=[]
        c=0
    else: sentence=sentence+[p]
    
#for p in tagged: print(p)
print('out-of-vocabulary words:',OOVcount)
print(occuracy(dataS,tagged))

