#!/usr/bin/env python3
import sys
import collections as col
# import icu
import math
import random
import datetime
import itertools
from optparse import OptionParser
from sys import stdin as sin
from sys import stdout as sout

STARTw = "<s>"
STARTt = "<s>"


def getprob(p, h):
    """ Returns a corresponding value from the given tuple if an item is in the tuple, otherwise returns zero."""
    if h in p:
        return p[h]
    else:
        return 0


def get_parametres_superv(data_wt):
    count = len(data_wt)
    tags = [t for (_, t) in data_wt]
    tags_uniq = col.Counter(tags)
    data_uniq = col.Counter([(w, t) for (w, t) in data_wt])
    p_wt = {(w, t): data_uniq[w, t] / tags_uniq[t] for (w, t) in
            data_uniq}  # todo: v slidech je psano v jinem poradi (tag,w), je to opravdu totez??????
    # p_wt=p_wt.union({(w,t):0 for w in {w for (w,_) in data_wt} for t in tags_uniq if (w,t) not in data_uniq })
    pc = [{(w, t): data_uniq[w, t] for (w, t) in data_uniq},
          tags_uniq]  # todo: v slidech je psano v jinem poradi (tag,w), je to opravdu totez??????
    p_t0 = 1 / len(tags_uniq)
    p_t1 = {t: tags_uniq[t] / count for t in tags_uniq}
    bigram_tags = col.Counter(zip(tags[:-1], tags[1:]))
    p_t2 = {(t1, t2): (bigram_tags[t1, t2] / tags_uniq[t1]) for (t1, t2) in bigram_tags}

    trigram_tags = col.Counter([t for t in zip(tags[:-2], tags[1:-1], tags[2:])])
    p_t3 = {(t1, t2, t3): (trigram_tags[t1, t2, t3] / bigram_tags[t1, t2]) for (t1, t2, t3) in trigram_tags}

    p_tt = [p_t0, p_t1, p_t2, p_t3]
    return (p_wt, p_tt, pc)


# ----------------- smoothing EM algorithm------------------------------------------

def EMiter(data, p, l):
    """An one iteration of EM algorithm."""
    tri = [u for u in zip(data[:-2], data[1:-1], data[2:])]
    pp = {
        (i, j, k): l[3] * getprob(p[3], (i, j, k)) + l[2] * getprob(p[2], (j, k)) + l[1] * getprob(p[1], k) + l[0] * p[
            0]
        for (i, j, k) in set(tri)}  # new p'(lambda)
    c = [0, 0, 0, 0]
    for (i, j, k) in tri:
        pptemp = pp[(i, j, k)]
        c[0] = c[0] + l[0] * p[0] / pptemp
        c[1] += l[1] * getprob(p[1], k) / pptemp
        c[2] += l[2] * getprob(p[2], (j, k)) / pptemp
        c[3] += l[3] * getprob(p[3], (i, j, k)) / pptemp
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
    sout.write("\nnumber of iterations:" + str(itercount) + ", precision: " + str(e) + "\n")
    return nextl


# ----------------------------------------------------------------------------------
def smoothEM(p, heldout, traindata):
    """
    Do linear interpolation trigram smoothing, need estimated probabilities form train data, heldout data fo estimate lambdas and testing data.
    """
    l = EMalgorithm(heldout, p)  # get lambdas
    tri = [i for i in zip(traindata[:-2], traindata[1:-1], traindata[2:])]
    # TODO: TO JE DOCELA DIVNE, ZE TU POUZIVAM TRIGRAMY MISTO VSECH MOZNYCH TROJIC (pozor, stejne tak v EMiter) , tady to nedává smysl, TODO: OVERIT, ZE MAM ZDE POUZIT TRAINING DATA
    ttrainset=set(traindata) #todo:pouzit ta globalni, pokud takto spravne
    pt_em = {
        (i, j, k): (
            l[0] * p[0] + l[1] * getprob(p[1], k) + l[2] * getprob(p[2], (j, k)) + l[3] * getprob(p[3], (i, j, k)))
        #for (i, j, k) in tri}
        for i in ttrainset for j in ttrainset for k in ttrainset}
    return pt_em


def smoothAdd1(pc, data, tagset):
    """
    Do smoothing by Adding 1, need counts of c(t,w) and c(h) from train data and need test data for smooting.
    """
    pwt = {(w, t): ((getprob(pc[0], (w, t)) + 1) / (getprob(pc[1], t) + len(wordset) * len(tagset))) for w in words for
           t in tagset}
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
        # DEBUG:
        print(getprob(self.wt_bigram_counts, (w, t)) + 1, " ", getprob(self.t_unigram_counts, t), " ", self.len_wordset,
              " ", self.len_tagset)
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



#def getword(word,)
def viterbi(text,tagset,wordset,Pwt,Ptt):

        if len(text)==0: return []
        V={}
        path={}
        isOOV=false # says if the proceeded word is out-of-vocabulary
        s0=(STARTt,STARTt)
        si=(STARTt,STARTt)
        t=''
        #V[-1,STARTt,STARTt]=1
        while STARTt in tagset: 
            # we dont want to have this tag in the middle of the tag sequence,  
            #for sure while instead of only remove
                 tagset.remove(STARTt) 
        w=text[0]
        V[0]={} # V has following format: V[time][state]=(probability,path)
        for t in tagset:
            V[0][STARTt,t]=(Ptt.get_ptt(t,STARTt,STARTt)*Pwt.get_pwt(w,t),[t])

        for k in range(1,len(text)-1):
                isOOV=false
                w=text[k]
                if w not in wordset: 
                    isOOV=true
                for t in tagset:
                    bests=''
                    bestpath=''
                    m=0 # maximum
                    for (i,j) in V[k-1]:
                        value=V[k-1][i,j][0]*Ptt.get_ptt(t,i,j)
                        if value>m: 
                            (bests[0], bests[1], m, bestpath)==(i, j, value, V[k-1][i,j][1]
)
                    V[k][bests[1],t]=(m*Pwt.get_pwt(w,t,isOOV),bestpath+[t]) 
#když si budu šikovně pamatovat cestu, tak si nepotřebuji pamatovat všech k stavů, ale jen vždy předchozí, tedy místo V[k,...] pouze V[0/1,...]
                   # V[k,j,t]=max([V[k-1],i,j)*Ptt.get_ptt(t,i,j) for (i,j) in V[k-1]])*Pwt.get_pwt(w,t,isOOV) chci něco v tomto smyslu, ale toto úplně nefunguje
                        # mám V[k-1,ti,tj]p
                        # chci V[k,tj,t]=max(V[k-1,ti,tj]*Ptt.get_ptt(t,ti,tj))*Pwt.get_pwt(w,t,isOOV) přes všechna ti,tj, která jsou v V[k-1]
                        #V[k,,t]
                        # a ještě path



        print("todo: the end")
        tagset = [STARTt] + tagset  # to be the same as at start

        return ""


# -----------------------------initialization-------------------------------

# ----- parsing arguments ---------
# if len(sys.argv) != 3 or (sys.argv[2][0] != "U" and sys.argv[2][0] != "S"):
#     sys.exit(
#         'Not correct arguments, please run with 2 arguments: input-file,\
#     number (1=task with words, 2=task with tags), count of resulting classes (1 for full hierarchy).')
# open text file in text read mode
parser = OptionParser(usage="usage: %prog [options] filename count")
parser.add_option("-s", "--supervised",
                  action="store_true", dest="supervised", default=False,
                  help="Use supervised method (the default is unsupervised)")
parser.add_option("-u", "--unsupervised",
                  action="store_false", dest="supervised", default=False,
                  help="Use unsupervised method (the default option)")
(options, args) = parser.parse_args()
file_name = args[0]
f = open(file_name, encoding="iso-8859-2", mode='rt')
supervised = options.supervised
# if sys.argv[2][0] == 'S':
#     supervised = True
# else:
#     supervised = False
# ------ data preparation ---------

data = [l.split('/', 1) for l in f.read().splitlines()]  # items in format: word,speech-tag which can contains '/'
dataT = [(STARTw, STARTt), (STARTw, STARTt)] + data[:60000]
dataH = [(STARTw, STARTt), (STARTw, STARTt)] + data[-60000:-40000]
dataS = [(STARTw, STARTt), (STARTw, STARTt)] + data[-40000:]
data = []  # for gc

tagsetT = set([t for (_, t) in dataT])
wordsetT = set([w for (w, _) in dataT])
# tag_set=set([t for (_,t) in dataH]+[t for (_,t) in dataT])

# ------- computation -------------

if supervised:
    pp = get_parametres_superv(dataT)  # get p_t a p_wt from test data, not smoothed yet
else:
    print("todo")
    sys.exit()
p_t = smoothEM(pp[1], [t for (_, t) in dataH], [t for (_, t) in dataT])  # probabilities p_t1,p_t2,p_t3
# todo: toto neni sikovne, zbytecne zabira moc pameti, lepsi neco typu getter a vzdy spocist
# p_wt=smoothAdd1(pp[2],[w for (w,_) in dataT],set([t for (_,t) in dataT]))
# #sem by šlo dát i dataH, resp. cele p_wt spocitat i z heldout - zabiralo hodne pameti
pwt = Pwt(pp[2][0], pp[2][1], len(wordsetT), len(tagsetT))
pt = Ptt(pp[1], [t for (_, t) in dataH], [t for (_, t) in dataT])
# viterbi(dataS,tagsetT, wordsetT) # zvlážit, zda nedat tagset a wordset i z heldout
# potřebuji p(t|u,v), p_wt(w/t) = c_wt(t,w)/c_t(t)
