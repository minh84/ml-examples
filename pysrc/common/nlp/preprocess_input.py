from collections import Counter
import os
import numpy as np
import pickle

PUNCTUATION_DICT = {'.' : ' <PERIOD> ',
                    ',' : ' <COMMA> ',
                    '"' : ' <QUOTATION_MASK> ',
                    ';' : ' <SEMICOLON> ',
                    '!' : ' <EXCLAMATION_MARK> ',
                    '?' : ' <QUESTION_MARK> ',
                    '(' : ' <LEFT_PAREN> ',
                    ')' : ' <RIGHT_PAREN> ',
                    '--': ' <HYPHENS> ',
                    ':' : ' <COLON> '}

def create_lookup_tables(word_counts, min_count = 0):
    """
    Create lookup tables for vocabulary and adding special <UNK> for unknown words
    :param word_counts: Input Counter object word -> count
    :param min_count: Ignore word that appears <= min_count
    :return:
        vocabs: all unique words
        word2id: a dict word -> integer id
        id2word: a dict integer id -> word
    """
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    # get vocabs, word2id, id2word
    vocabs = [w for w in sorted_vocab if word_counts[w] > min_count]
    word2id = {word: ii for ii, word in enumerate(vocabs)}
    id2word = {ii: word for word, ii in word2id.items()}

    return vocabs, word2id, id2word

class Word2VecInput(object):
    '''
    Pre-processing input raw text
        .) read text and replace punctuation with special tokens
        .) do the pre-process
    '''
    def __init__(self,
                 filename):
        assert os.path.isfile(filename)

        self.path = filename

        # read_data
        self._words   = None
        self.read_data()

        # pre-processing data: create vocab, word2id, id2word then sub-sampling data
        self._vocabs  = None
        self._word2id = None
        self._id2word = None

        self._freqs         = None
        self._reject_probs  = None
        self._train_wordids = None    # training word-ids
        self.preprocess_data()

    def read_data(self, reset = False):
        if self._words is None or reset:
            with open(self.path, 'r') as f:
                text = f.read()

            # replace punctuation with tokens
            for k,v in PUNCTUATION_DICT.items():
                text = text.replace(k, v)

            self._words = text.split()

    def preprocess_data(self, reset = False):
        if self._vocabs is None or reset:
            # sub-sampling words
            #       i)  remove words which appear <= 5 times (e.g typos, rare words)
            #       ii) sub-sampling too frequent words such as 'the,a,an,...,e.t.c'
            #           as described in https://arxiv.org/pdf/1301.3781.pdf
            #           we reject word with prob 1 - sqrt(t/freq(w)), t=1e-5
            assert (self._words != None)
            counter = Counter(self._words)
            min_count = 5

            self._vocabs, self._word2id, self._id2word = create_lookup_tables(counter, min_count = min_count)

            # trimmed down word appear <= 5
            trimmed_words = [w for w in self._words if w in self._word2id]

            # compute word frequence
            total_count = len(trimmed_words)
            self._freqs = {w : counter[w]/total_count for w in self._vocabs}

            # compute word reject_probs
            threshold = 1.0e-5
            self._reject_probs ={w : max(0., 1.0 - np.sqrt(threshold/self._freqs[w])) for w in self._vocabs}

            # sub-sampling training words
            self._train_wordids = [self._word2id[w] for w in trimmed_words if self._reject_probs[w] < np.random.random()]

    def dump(self, outfile):
        pickle.dump((self._vocabs,
                     self._word2id,
                     self._id2word,
                     self._freqs,
                     self._train_wordids), open(outfile, 'wb'))

