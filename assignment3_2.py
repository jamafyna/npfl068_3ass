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
from classes import LinearSmoothedDistribution
from classes import AddOneSmoothedDistribution
from functions import baum_welch


STARTw = "###" # start token/token which split sentences
STARTt = "###" # STARTw for words, STARTt for tags


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
    return p_tt, set(trigram_tags)




def smoothAdd(pc, data, tagset, lamb=2**(-10)):
    """
    Do smoothing by Adding less than 1, need counts of c(t,w) and c(h) from train data and need test data for smooting.
    """
    pwt = {(w, t): ((get_prob(pc[0], (w, t)) + lamb) / (get_prob(pc[1], t) + lamb*len(wordsetT) * len(tagset))) for w in wordsetT for t in tagset}
    return pwt

#------------------ ADD SMOOTHING -----------------------------------

class Pwt:
    """
    Computes the initial distribution of p(w|t) and smooths it.
    """
    wt_counts = []
    t_counts = []
    vocab_size=0
   

    def __init__(self,data,wordset_size,tagset_size):
        self.t_counts = Counter([t for (_,t) in data])
        self.wt_counts = Counter(data)  # Counter([(w, t) for (w, t) in data_wt])
        self.vocab_size=wordset_size*tagset_size
        self.tagset_len=tagset_size

    def get_smoothed_pwt(self, w, t, isOOV, lamb=2**(-10)):
        """
        Returns smoothed p(w|t), by smoothing less than 1. Suppose that w and t are from known wordset and tagset, not unknown.
        """
             # if the w is out-of-vocabulary, then use uniform distribution
        if isOOV: return 1/self.tagset_len
        return (self.wt_counts[w,t]+lamb)/(self.t_counts[t]+lamb*self.vocab_size)


#--------------- EM SMOOTHING ----------------------------------------------

class Ptt:
    """
    Class for getting smoothed arc probability.  Do linear interpolation trigram smoothing, need estimated probabilities form train data, heldout data fo estimate lambdas and testing data.
    """
    pl={}

    def __init__(self, p, heldout, train, less_memory=False):
        print("\n---Smoothing, EM algorithm:---\n")
        self.memory=less_memory
        self.l = self.EMalgorithm(heldout, p)  # computes lambdas
        print("lambdas:\n l0:",self.l[0],"\nl1: ",self.l[1],"\nl2: ",self.l[2],"\nl3: ",self.l[3],"\n")
        if(not less_memory): self.p_t = self.compute_full(p, heldout, train)  
        self.p=p
        self.pl[0]=self.p[0]*self.l[0]
        for i in range(1,4):
            self.pl[i]={t : (self.p[i][t]*self.l[i]) for t in p[i]}
            

    def get_ptt(self, t1, t2, t3):
        """
        Returns smoothed p(t3|t1,t2).
        """
        if self.memory:
              return self.l[0] * self.p[0] + self.l[1] * get_prob(self.p[1], t3) + self.l[2] * get_prob(self.p[2], (t2, t3)) + self.l[3] * get_prob(self.p[3], (t1, t2, t3))

        else: return self.p_t[t1, t2, t3]
    
    def get_prec_ptt(self, t1, t2, t3):
        if self.memory:
                return self.pl[0] + get_prob(self.pl[1], t3) + get_prob(self.pl[2], (t2, t3)) + get_prob(self.pl[3], (t1, t2, t3))
        else: return self.p_t[t1, t2, t3]
        
         
    def compute_full(self, p, heldout, traindata):
              
        tri = [i for i in zip(traindata[:-2], traindata[1:-1], traindata[2:])]
        ttrainset=set(traindata) 
        pt_em = {
            (i, j, k): (
                self.l[0] * p[0] + self.l[1] * get_prob(p[1], k) + self.l[2] * get_prob(p[2], (j, k)) + self.l[3] * get_prob(p[3], (i, j, k)))
            for i in ttrainset for j in ttrainset for k in ttrainset}
        return pt_em


    def EMiter(self,data, p, l):
        """An one iteration of EM algorithm."""
        tri = [u for u in zip(data[:-2], data[1:-1], data[2:])]
        pp = {
            (i, j, k): l[3] * get_prob(p[3], (i, j, k)) + l[2] * get_prob(p[2], (j, k)) + l[1] * get_prob(p[1], k) + l[0] * p[0] for (i, j, k) in set(tri)
             }  # new p'(lambda)
        c = [0, 0, 0, 0]
        for (i, j, k) in tri:
            pptemp = pp[(i, j, k)]
            c[0] += l[0] * p[0] / pptemp
            c[1] += l[1] * get_prob(p[1], k) / pptemp
            c[2] += l[2] * get_prob(p[2], (j, k)) / pptemp
            c[3] += l[3] * get_prob(p[3], (i, j, k)) / pptemp
        return [i / sum(c) for i in c]  # normalised


    def EMalgorithm(self,data, p):  # heldoutdata
        """EM algorithm, input: data: heldout data, p: probabilities counted form training data """
        l = [10, 10, 10, 10]  # infinity, due to first while
        nextl = [0.25, 0.25, 0.25, 0.25]  # lambdas counted in the next iteration
        e = 0.001  # precision
        itercount = 0
        while (abs(l[0] - nextl[0]) >= e or abs(l[1] - nextl[1]) >= e or abs(l[2] - nextl[2]) >= e or abs(
                    l[3] - nextl[3]) >= e):  # expected precision is not yet achieved
            l = nextl
            nextl = self.EMiter(data, p, l)
            itercount = itercount + 1
        sout.write("\nnumber of iterations:" + str(itercount) + ", precision: " + str(e) + "\n")
        return nextl


# ------------------------------------------------------------------------------
# ---------------------PUVODNI VITERBI ----------------------------------------

def viterbi_fast(text,tagset,wordset,trigramtagset,Pwt,Ptt,start):
        """
        Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset), and first half of a start state (usually end of previous sentence or start token).
        """
        if len(text)==0: return []
        isOOV=False # indicates if proceeded word is out-of-vocabulary
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
        V[0][start]=(1,[start])
        V[1][STARTt,STARTt]=(1,[start,(STARTw,STARTt)])
        now=1 # value depends on k which starts from 1
        prev=0
        treshold=0.00000000000000001
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
                    if t==STARTt: continue
                    bests={}
                    bestpath=[]
                    maxprob=0
                    for (i,j) in V[prev]:
                        value=V[prev][i,j][0]*Ptt.get_ptt(i,j,t)
                        if value>=maxprob: # '=' because of very small numbers  
                            bests[0]=i
                            bests[1]=j
                            maxprob=value
                            bestpath=V[prev][i,j][1]
                    V[now][bests[1],t]=(maxprob*(Pwt.get_smoothed_pwt(w,t,isOOV)),bestpath+[(w,t)])
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


def viterbi2(text,tagset,wordset,trigramtagset,Pwt,Ptt,start):
        """
        Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset), and first half of a start state (usually end of previous sentence or start token).
        """
        if len(text)==0: return []
        isOOV=False # indicates if proceeded word is out-of-vocabulary
        if text[0]==STARTw: 
            warnings.warn("inconsistend data, start tokens",Warning)
        else: text=[(STARTw, STARTt)]+text 
        V={}    # structure for remember viterbi computation
                # V[time][state]=(probability,path) where state==(tag1,tag2)
        V[0]={} 
        V[1]={}
        OOVcount=0
        # --- initialisation, starting state
        V[0][start]=(0,[start])
        V[1][STARTt,STARTt]=(1,[start,(STARTw,STARTt)])
        now=1 # value depends on k which starts from 1
        prev=0
        treshold=0.00000000000000001
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
                for (i,j,t) in trigramtagset:
                    #partprob=Ptt.pl[0]+get_prob(Ptt.pl[1],t)
                    if (i,j) not in V[prev]: continue
                    value=V[prev][i,j][0]*Ptt.get_ptt(i,j,t)
                    #value=V[prev][i,j][0]*(partprob+get_prob(Ptt.pl[2],(j,t))+get_prob(Ptt.pl[3],(i,j,t)))
                    if ((j,t) not in V[now]) or value>V[now][j,t][0]: # '=' because of very small numbers              
                            V[now][j,t]=(value*Pwt.get_smoothed_pwt(w,t,isOOV),
                                         V[prev][i,j][1]+[(w,t)])
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

# ----------------- VITERBI ----------------------------------------------

def viterbi(text,tagset,wordset,trigramtagset,Pwt,Ptt,start):
        """
        Assign the most probably tag sequence to a given sentence 'text'. Needs set of tags (tagset), vocabulary (wordset), and first half of a start state (usually end of previous sentence or start token).
        """
        if len(text)==0: return []
        isOOV=False # indicates if proceeded word is out-of-vocabulary
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
        V[0][start]=(0,[start])
        V[1][STARTt,STARTt]=(1,[start,(STARTw,STARTt)])
        now=1 # value depends on k which starts from 1
        prev=0
        treshold=0.00000000000000001
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
                    if t==STARTt: continue
                    bests={}
                    bestpath=[]
                    maxprob=0
                    #partprob=Ptt.pl[0]+get_prob(Ptt.pl[1],t)
                    for (i,j) in V[prev]:
                        if(i,j,t) not in trigramtagset: continue
                        value=V[prev][i,j][0]*Ptt.get_ptt(i,j,t)
                        #value=V[prev][i,j][0]*(partprob+get_prob(Ptt.pl[2],(j,t))+get_prob(Ptt.pl[3],(i,j,t)))
                        if ((j,t) not in V[now]) or value>V[now][j,t][0]: # '=' because of very small numbers              
                            V[now][j,t]=(value*Pwt.get_smoothed_pwt(w,t,isOOV),
                                         V[prev][i,j][1]+[(w,t)])
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


# ------------------- EVALUATION ------------------------------------------
def occuracy(right,computed):
       if len(right)!=len(computed): 
            warnings.warn("inconsistend data, different length of test and computed solution",Warning)
       correct=0
       allc=min(len(right),len(computed))
       for i in range(0,allc-1):
           if right[i][0]!=computed[i][0]:
               raise Exception("inconsistent data, different words in test and computed")
           if right[i][1]==computed[i][1]: correct+=1
       print("right tagged: ",correct,", all: ",allc)
       return (correct/allc)







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
(options, args) = parser.parse_args()
file_name = args[0]
file = open(file_name, encoding="iso-8859-2", mode='rt')
supervised = options.supervised
memory = options.memory

# ------ data preparation ---------

data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))
dataT = [(STARTw, STARTt), (STARTw, STARTt)] + data[:-60000]  # training data
dataH = [(STARTw, STARTt), (STARTw, STARTt)] + data[-60000:-40000]  # held_out data 
#dataS = data[-40000:]  # testing data
dataS =  data[-4000:]  # testing data
if dataS[0]!=[(STARTw,STARTt)]: dataS= [(STARTw,STARTt)]+dataS
data = []  # for gc
OOVcount=0 # unknown words
tagsetT = set([t for (_, t) in dataT]) # set of tages
wordsetT = set([w for (w, _) in dataT]) #set of words

# ------- computation -------------

if supervised:
    pp, trig_tag_set = get_parametres_superv([t for (_,t) in dataT]) 
         # get distribution of tags from train data, not smoothed yet
else:
    # Baum-Welch training
    transition_matrix, emission_matrix = baum_welch(dataT, dataH)
    # save the trained model
    np.save('transition_matrix.npy', transition_matrix)
    np.save('emission_matrix.npy', emission_matrix)
    sys.exit()

#sem by šlo dát i dataH, resp. cele p_wt spocitat i z heldout - zabiralo hodne pameti
pwt = Pwt(dataT,len(wordsetT),len(tagsetT))
         # distribution of pairs (word,tag),smoothed

pt = Ptt(pp, [t for (_, t) in dataH], [t for (_, t) in dataT],memory)
         #distrib. of tags, smoothed

# -------- tagging by Viterbi----------------
tagged=[]
sentence=[]
v=[]
c=0
sentence_end=(STARTw,STARTt)
# run Viterbi on each sentence of the test data
print("--- Starting Viterbi --- ")
for p in dataS:
    if p==(STARTw,STARTt): 
        if(sentence!=[]):
            #v,c=viterbi_fast([w for (w,_) in sentence], tagsetT, wordsetT, trig_tag_set, pwt, pt, sentence_end)
            v,c=viterbi([w for (w,_) in sentence], tagsetT, wordsetT, trig_tag_set, pwt, pt, sentence_end)
        tagged=tagged+v+[(STARTt,STARTw)]
        for t in tagged: print(t)
        OOVcount+=c
        if len(sentence)==0: sentence_end=(STARTw,STARTt)
        else: sentence_end=sentence[len(sentence)-1]
        sentence=[]
        v=[]
        c=0
    else: sentence=sentence+[p]
   
print("---VITERBI ENDED ---")    
print('out-of-vocabulary words:',OOVcount)
o=occuracy(dataS,tagged)
print('occuracy:', o)
