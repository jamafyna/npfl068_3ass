#!/usr/bin/env python3
from optparse import OptionParser

# import Brill's tagger from NLTK
from nltk.tag import brill, brill_trainer, UnigramTagger


def train_brill_tagger(train_data, **kwargs):
    # use the initial rules from the NLTK demonstration
    templates = brill.nltkdemo18()
    # use unigram tagger as the initial tagger
    initial_tagger = UnigramTagger(train_data)
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
    return trainer.train(train_data, **kwargs)


# parse command lines arguments
parser = OptionParser(usage="usage: %prog filename")
(options, args) = parser.parse_args()
file_name = args[0]
# read the data from the file
file = open(file_name, 'rt', encoding='latin2')
data = []
for line in file:
    w, t = line.strip().split(sep='/', maxsplit=1)
    data.append((w, t))

# the initial split from the assignment
testing = data[-40000:]
training = data[:-60000]
print('The accuracy obtained by the 1st split is:')
tagger = train_brill_tagger([training])
print(tagger.evaluate([testing]))

# the second split from the assignment
# held-out is 40000-60000
testing = data[:40000]
training = data[60000:]
print('The accuracy obtained by the 2nd split is:')
tagger = train_brill_tagger([training])
print(tagger.evaluate([testing]))

# held-out is 60000-80000
testing = data[80000:120000]
training = data[:6000] + data[120000:]
print('The accuracy obtained by the 3rd split is:')
tagger = train_brill_tagger([training])
print(tagger.evaluate([testing]))

# held-out is 160000-180000
testing = data[120000:160000]
training = data[:120000] + data[180000:]
print('The accuracy obtained by the 4th split is:')
tagger = train_brill_tagger([training])
print(tagger.evaluate([testing]))

# use the held outs from second and third as the training data
testing = data[40000:80000]
training = data[:40000] + data[100000:]
print('The accuracy obtained by the 5th split is:')
tagger = train_brill_tagger([training])
print(tagger.evaluate([testing]))
