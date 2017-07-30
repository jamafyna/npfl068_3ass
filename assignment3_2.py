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

STARTw = "###"
STARTt = "###"


def get_prob(p, h):
    """ Returns a corresponding value from the given tuple if an item is in the tuple, otherwise returns zero."""
    if h in p:
        return p[h]
    else:
        return 0


def get_parametres_superv(data_wt):
    count = len(data_wt)
    tags = [t for (_, t) in data_wt]
    tags_uniq = Counter(tags)
    data_uniq = Counter(data_wt)  # Counter([(w, t) for (w, t) in data_wt])
    p_wt = {(w, t): data_uniq[w, t] / tags_uniq[t] for (w, t) in data_uniq}
    # todo: v slidech je psano v jinem poradi (tag,w), je to opravdu totez??????
    # @ add todo: naopak je to len preto, lebo P(vugenerujem w | viem t), ale v dictionary je to jedno, kedze hladas tu pravdepodobnost ale neviem naisto
    # p_wt=p_wt.union({(w,t):0 for w in {w for (w,_) in data_wt} for t in tags_uniq if (w,t) not in data_uniq })
    pc = [{(w, t): data_uniq[w, t] for (w, t) in data_uniq}, tags_uniq]
    # constant probability mass (avoiding zeros)
    p_t0 = 1 / len(tags_uniq)
    # unigram probabilities
    p_t1 = {t: tags_uniq[t] / count for t in tags_uniq}
    # bigram probabilities
    bigram_tags = Counter(zip(tags[:-1], tags[1:]))
    p_t2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}
    # trigram probabilities
    trigram_tags = Counter([trig for trig in zip(tags[:-2], tags[1:-1], tags[2:])])
    p_t3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}
    p_tt = [p_t0, p_t1, p_t2, p_t3]
    return p_wt, p_tt, pc


# ----------------- smoothing EM algorithm------------------------------------------

def EMiter(data, p, l):
    """An one iteration of EM algorithm."""
    tri = [u for u in zip(data[:-2], data[1:-1], data[2:])]
    pp = {
        (i, j, k): l[3] * get_prob(p[3], (i, j, k)) + l[2] * get_prob(p[2], (j, k)) + l[1] * get_prob(p[1], k) + l[0] *
                                                                                                                 p[
                                                                                                                     0]
        for (i, j, k) in set(tri)}  # new p'(lambda)
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
    print(l)
    tri = [i for i in zip(traindata[:-2], traindata[1:-1], traindata[2:])]
    ttrainset=set(traindata) 
    pt_em = {
        (i, j, k): (
            l[0] * p[0] + l[1] * get_prob(p[1], k) + l[2] * get_prob(p[2], (j, k)) + l[3] * get_prob(p[3], (i, j, k)))
        # for (i, j, k) in tri}
        for i in ttrainset for j in ttrainset for k in ttrainset}
    return pt_em


def smoothAdd1(pc, data, tagset):
    """
    Do smoothing by Adding 1, need counts of c(t,w) and c(h) from train data and need test data for smooting.
    """
    pwt = {(w, t): ((getprob(pc[0], (w, t)) + 1) / (getprob(pc[1], t) + len(wordsetT) * len(tagset))) for w in wordsetT for t in tagset}
    return pwt


class Pwt:
    wt_bigram_counts = []
    t_unigram_counts = []
    len_wordset = 1
    len_tagset = 1

    def __init__(self, wtcounts, tcounts, wlen, tlen):
        self.wt_bigram_counts = wtcounts
        self.t_unigram_counts = tcounts
        self.len_wordset = wlen
        self.len_tagset = tlen

    def get_pwt(self, w, t, isOOV=False):
        """
        Returns smoothed p(w|t). Suppose that w and t are from known wordset and tagset, not unknown.
        """
        if isOOV: return 1/self.len_tagset # if the w is out-of-vocabulary, then use uniform distribution
        return ((getprob(self.wt_bigram_counts, (w, t)) + 1) / (
           getprob(self.t_unigram_counts, t) + self.len_wordset * self.len_tagset))

class Ptt:
    """
    Class for getting smoothed arc probability
    """
    # TODO: ČASEM možná vylepšit na nepamatování si celé tabulky, ale dynam.počítání
    p_t = []

    def __init__(self, pp, heldout, train):
        self.p_t = smoothEM(pp, heldout, train)  # probabilities p_t1,p_t2,p_t3

    def get_ptt(self, t1, t2, t3):
        """
        Returns smoothed p(t3|t1,t2).
        """
        return self.p_t[t3, t1, t2]  # TODO: Možná udělat časem dynamicky


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
                        value=V[prev][i,j][0]+math.log(Ptt.get_ptt(t,i,j),2)
                        if value>=maxprob: # '=' because of very small numbers  
                            bests[0]=i
                            bests[1]=j
                            maxprob=value
                            bestpath=V[prev][i,j][1]
                    V[now][bests[1],t]=(maxprob+math.log(Pwt.get_pwt(w,t,isOOV),2),bestpath+[(w,t)])
        if tagsetcontainsSTART: tagset.add(STARTt)  # to be the same as at start
        # --- final search the best tag sequence
        maxprob=-float('inf')
        ends={}              # the best end state
        for s in V[now]:
                if V[now][s][0]>=maxprob:
                    maxprob=V[now][s][0]
                    ends=s
        return V[now][ends][1][2:],OOVcount # only [2:] because of start tokens

def viterbi(text,tagset,wordset,Pwt,Ptt):
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
        OOVcount=0
        # --- initialisation, starting state
        V[1][STARTt,STARTt]=(1,[[STARTw,STARTt],[STARTw,STARTt]])
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
                    maxprob=0
                    for (i,j) in V[prev]:
                        value=V[prev][i,j][0]*Ptt.get_ptt(t,i,j)
                       # if value==0: Chtělo by to aspoň takovýto treshold, zahodit všechny stavy mající value=0, což může být velmi velmi malé číslo zaokrouhlené na 0
                        if value>=maxprob: # '=' because of very small numbers  
                            bests[0]=i
                            bests[1]=j
                            maxprob=value
                            bestpath=V[prev][i,j][1]
                    V[now][bests[1],t]=(maxprob*Pwt.get_pwt(w,t,isOOV),bestpath+[[w,t]])
        if tagsetcontainsSTART: tagset.add(STARTt)  # to be the same as at start
        # --- final search the best tag sequence
        maxprob=0
        ends={}              # the best end state
        for s in V[now]:
                if V[now][s][0]>=maxprob:
                    maxprob=V[now][s][0]
                    ends=s
        if maxprob==0: warnings.warn("zero maxprobability")
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

#data = [l.split('/', 1) for l in f.read().splitlines()]  # items in format: word,speech-tag which can contains '/'
#dataT = [[STARTw, STARTt], [STARTw, STARTt]] + data[:-60000]
#dataH = [[STARTw, STARTt], [STARTw, STARTt]] + data[-60000:-40000]
#dataS = data[-40000:] # the right testing data
# @ prerobil som to aby tam boli rovno tuple a bolo to potom jednoduchsie
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))
# @ tu bola chyba, trenovacie data mali iba 60000 vzorov, ale ich ma byt ten zvysok --- pridane minusko
dataT = [(STARTw, STARTt), (STARTw, STARTt)] + data[:-60000]  # training data
dataH = [(STARTw, STARTt), (STARTw, STARTt)] + data[-60000:-40000]  # held_out data used for smoothing
dataS = [(STARTw, STARTt), (STARTw, STARTt)] + data[-40000:]  # testing data
#dataS = data[-39:] # testingdata for debuging
if dataS[0]!=[[STARTw,STARTt]]: dataS= [[STARTw,STARTt]]+dataS
data = []  # for gc
OOVcount=0
tagsetT = set([t for (_, t) in dataT])
wordsetT = set([w for (w, _) in dataT])

# ------- computation -------------

if supervised:
    pp = get_parametres_superv(dataT)  # get p_t a p_wt from test data, not smoothed yet
else:
    print("todo")
    sys.exit()
#p_t = smoothEM(pp[1], [t for (_, t) in dataH], [t for (_, t) in dataT])  # probabilities p_t1,p_t2,p_t3
p_wt=smoothAdd1(pp[2],[w for (w,_) in dataT],set([t for (_,t) in dataT]))
#sem by šlo dát i dataH, resp. cele p_wt spocitat i z heldout - zabiralo hodne pameti
pwt = Pwt(pp[2][0], pp[2][1], len(wordsetT), len(tagsetT))
pt = Ptt(pp[1], [t for (_, t) in dataH], [t for (_, t) in dataT])


# -------- tagging ----------------
tagged=[]
sentence=[]
v=[]
c=0
for p in dataS:
    if p==['###','###']: 
        if(sentence!=[]):
            v,c=viterbi([w for (w,_) in sentence], tagsetT,wordsetT,pwt,pt)
        tagged=tagged+v+[['###','###']]
        OOVcount+=c
        sentence=[]
        v=[]
        c=0
    else: sentence=sentence+[p]
    
#for p in tagged: print(p)
print('out-of-vocabulary words:',OOVcount)
print(occuracy(dataS,tagged))
