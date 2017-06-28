
import sys
import collections as col
import icu
import math
import random
import datetime
import itertools

sin=sys.stdin
sout=sys.stdout 





def getprob(p,h):
        """ Returns a corresponding value from the given tuple if an item is in the tuple, otherwise returns zero."""
        if h in p: 
            return p[h]
        else: return 0 

def get_parametres_superv(data_wt):
        count=len(data_wt)
        tags=[t for (_,t) in data_wt]
        tags_uniq=col.Counter(tags)
        data_uniq=col.Counter([(w,t) for (w,t) in data_wt])
        p_wt={(w,t) : data_uniq[w,t]/tags_uniq[t] for (w,t) in data_uniq}#todo: v slidech je psano v jinem poradi (tag,w), je to opravdu totez??????
       # p_wt=p_wt.union({(w,t):0 for w in {w for (w,_) in data_wt} for t in tags_uniq if (w,t) not in data_uniq }) nevim, jestli je potreba davat nuly an misto, ktere opravdu neexistuje, asi spis neni vhodne
        
        pc=[{(w,t) : data_uniq[w,t] for (w,t) in data_uniq}, tags_uniq]#todo: v slidech je psano v jinem poradi (tag,w), je to opravdu totez??????
        

        p_t0=1/len(tags_uniq)
        p_t1={t : tags_uniq[t]/count for t in tags_uniq}
        bigram_tags=col.Counter(zip(tags[:-1],tags[1:]))
        p_t2={(t1,t2): (bigram_tags[t1,t2]/tags_uniq[t1]) for (t1, t2) in bigram_tags}
        
        trigram_tags=col.Counter([t for t in zip(tags[:-2],tags[1:-1],tags[2:])])
        p_t3={(t1,t2,t3): (trigram_tags[t1,t2,t3]/bigram_tags[t1,t2]) for (t1,t2,t3) in trigram_tags}

        p_tt=[p_t0,p_t1,p_t2,p_t3]
        return (p_wt,p_tt,pc)
#----------------- smoothing EM algorithm------------------------------------------

def EMiter(data,p,l, big, tri):
        """An one iteration of EM algorithm."""
        pp={(i,j,k):l[3]*getprob(p[3],(i,j,k))+l[2]*getprob(p[2],(j,k))+l[1]*getprob(p[1],k)+l[0]*p[0]  for (i,j,k) in set(tri)} # new p'(lambda)
        c=[0,0,0,0]
        for (i,j,k) in tri:
            pptemp=pp[(i,j,k)]
            c[0]=c[0]+l[0]*p[0]/pptemp 
            c[1]=c[1]+l[1]*getprob(p[1],k)/pptemp
            c[2]=c[2]+l[2]*getprob(p[2],(j,k))/pptemp
            c[3]=c[3]+l[3]*getprob(p[3],(i,j,k))/pptemp
        return [i/sum(c) for i in c] # normalised

def EMalgorithm(data, p): # heldoutdata
        """EM algorithm, input: data: heldout data, p: probabilities counted form training data """
        bigrams=[u for u in zip(data[:-1],data[1:])]
        trigrams=[u for u in zip(data[:-2],data[1:-1],data[2:])]
        l=[10,10,10,10] #infinity, due to first while 
        nextl=[0.25, 0.25, 0.25, 0.25] # lambdas counted in the next iteration 
        e=0.001 # precision
        itercount=0
        while (abs(l[0]-nextl[0])>=e or abs(l[1]-nextl[1])>=e or abs(l[2]-nextl[2])>=e or abs(l[3]-nextl[3])>=e): # expected precision is not yet achieved
                l=nextl
                nextl=EMiter(data,p,l,bigrams,trigrams)
                itercount=itercount+1
        sout.write("\nnumber of iterations:"+str(itercount)+", precision: "+str(e)+"\n")
        return nextl          
#----------------------------------------------------------------------------------
def smoothEM(p,heldout,data):
        """
        Do linear interpolation trigram smoothing, need estimated probabilities form train data, heldout data fo estimate lambdas and testing data. 
        """
        l=EMalgorithm(heldout,p)
        tri=[i for i in zip(data[:-2],data[1:-1],data[2:])] 
        pt={(i,j,k) : (l[0]*p[0]+l[1]*getprob(p[1],k)+l[2]*getprob(p[2],(j,k))+l[3]*getprob(p[3],(i,j,k))) for (i,j,k) in tri}
        return pt

def smoothAdd1(pc,data,tagset):
        """
        Do smoothing by Adding 1, need counts of c(t,w) and c(h) from train data and need test data for smooting.
        """
        words=col.Counter(data)
        pwt={(w,t) : ((getprob(pc[0],(w,t))+1)/(getprob(pc[1],t)+len(words)*len(tagset))) for w in words for t in tagset }
        return pwt



#-----------------------------initialization-------------------------------

if(len(sys.argv)!=4 or (sys.argv[2]!="1" and sys.argv[2]!="2") or not sys.argv[3].isdigit()): 
    sys.exit('Not correct arguments, please run with 3 arguments: input-file, number (1=task with words, 2=task with tags), count of resulting classes (1 for full hierarchy).')
f=open(sys.argv[1],encoding="iso-8859-2",mode='rt')
task=sys.argv[2]
finalcount=int(sys.argv[3])
data=[l.split('/',1) for l in f.read().splitlines()]  # items in format: word,speech-tag which can contains '/'
dataT=data[:60000]
dataH=data[-60000:-40000]
dataS=data[-40000:]

dataSwords=[t for (_,t) in dataS]

data_words=[word for (word,tag) in data]
data_tags=[tag for (word,tag) in data]
data=[] # for gc


pp=get_parametres_superv(dataT)
p_t=smoothEM(pp[1],[t for (_,t) in dataH],dataSwords) # probabilities p_t1,p_t2,p_t3
p_wt=smoothAdd1(pp[2],[t for (_,t) in dataT],dataSwords)
# potřebuji p(t|u,v), p_wt(w/t) = c_wt(t,w)/c_t(t)
