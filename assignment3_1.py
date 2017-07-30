#!/usr/bin/env python3
from optparse import OptionParser

# import Brill's tagger from NLTK
from nltk.tag import brill, brill_trainer, UnigramTagger


def train_brill_tagger(train_data, **kwargs):
    """Trains the Brill's tagger on the given data with the initial rules from the demo and initial unigram tagger"""
    # use the initial rules from the NLTK demonstration
    templates = brill.nltkdemo18()
    # use unigram tagger as the initial tagger
    initial_tagger = UnigramTagger(train_data)
    trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True)
    return trainer.train(train_data, **kwargs)


def split_sentences(data_to_split):
    """Splits the data into sentences and returns a list of lists of word-tag tuples"""
    sentences = []
    sentence = []
    for word, tag in data_to_split:
        if word != '###':
            sentence.append((word, tag))
        else:
            if sentence:
                sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


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
testing_sentences = split_sentences(testing)
training = data[:-60000]
training_sentences = split_sentences(training)
print('The accuracy obtained by the 1st split is:')
tagger = train_brill_tagger(training_sentences)
print(tagger.evaluate(testing_sentences))

# the second split from the assignment
# held-out is 40000-60000
testing = data[:40000]
testing_sentences = split_sentences(testing)
training = data[60000:]
training_sentences = split_sentences(training)
print('The accuracy obtained by the 2nd split is:')
tagger = train_brill_tagger(training_sentences)
print(tagger.evaluate(testing_sentences))

# held-out is 60000-80000
testing = data[80000:120000]
testing_sentences = split_sentences(testing)
training = data[:6000] + data[120000:]
training_sentences = split_sentences(training)
print('The accuracy obtained by the 3rd split is:')
tagger = train_brill_tagger(training_sentences)
print(tagger.evaluate(testing_sentences))

# held-out is 160000-180000
testing = data[120000:160000]
testing_sentences = split_sentences(testing)
training = data[:120000] + data[180000:]
training_sentences = split_sentences(training)
print('The accuracy obtained by the 4th split is:')
tagger = train_brill_tagger(training_sentences)
print(tagger.evaluate(testing_sentences))

# use the held outs from second and third as the training data
testing = data[40000:80000]
testing_sentences = split_sentences(testing)
training = data[:40000] + data[100000:]
training_sentences = split_sentences(training)
print('The accuracy obtained by the 5th split is:')
tagger = train_brill_tagger(training_sentences)
print(tagger.evaluate(testing_sentences))
