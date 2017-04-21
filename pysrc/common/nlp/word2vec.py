import numpy as np
import tensorflow as tf
from time import time

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])

    return list(target_words)


class Word2vecNeg(object):
    def __init__(self,
                 vocabs,
                 word2id,
                 id2word,
                 freqs,
                 train_wordids,
                 settings = {'embed_dim'     : 200,
                             'batch_size'    : 16,
                             'nb_neg_sample' : 100,
                             'window_size'   : 5}):
        self._vocabs = vocabs
        self._word2id = word2id
        self._id2word = id2word
        self._freqs = freqs

        self._train_wordids = train_wordids

        self._vocabs_size = len(self._vocabs)
        self._freqs_list = [0.] * self._vocabs_size
        for w in self._vocabs:
            self._freqs_list[self._word2id[w]] = self._freqs[w]

        # get hyper-parameters
        self._embed_dim     = settings.get('embed_dim',     200)
        self._batch_size    = settings.get('batch_size',    16)
        self._nb_neg_sample = settings.get('nb_neg_sample', 100)
        self._window_size   = settings.get('window_size',   5)


    #     self._neg_table_size = settings.get('neg_table_size', 1000000)
    #     self.negSampleTable()
    #
    # def negSampleTable(self, table_size = None):
    #     '''
    #     we build a sample table st where
    #         number of i such that st[i] = w is freq[w]
    #     this sample-table allow us to quickly do the nagative-sampling in the cost of memory
    #     :param table_size: size of sample-table
    #     :return: a sample table
    #     '''
    #     if table_size is None:
    #         table_size = self._neg_table_size
    #
    #     if self._sampleTable is None or len(self._sampleTable) != table_size:
    #         nTokens = len(self._id2word)
    #         samplingFreq = np.zeros((nTokens,))
    #
    #         for i,w in self._id2word.items():
    #             if w in self._freqs:
    #                 freq = self._freqs[w] ** 0.75
    #             else:
    #                 freq = 0.0
    #             samplingFreq[i] = freq
    #
    #         samplingFreq /= np.sum(samplingFreq)
    #         samplingFreq = np.cumsum(samplingFreq) * table_size
    #
    #         self._sampleTable = [0] * table_size
    #
    #         j = 0
    #         for i in range(table_size):
    #             while i > samplingFreq[j]:
    #                 j += 1
    #             self._sampleTable[i] = j
    #
    #     return self._sampleTable
    #
    # def negSampleIdx(self):
    #     return self.negSampleTable()[np.random.randint(0, self._neg_table_size-1)]

    def _create_placeholder(self):
        with self._graph.as_default():
            # first dim is batch-size can be variable so we set it to None
            self._center_words = tf.placeholder(tf.int32, shape=[None],    name='center_words')
            self._target_words = tf.placeholder(tf.int32, shape=[None, 1], name='target_words')

    def _create_embedding(self):
        with self._graph.as_default():
            with tf.name_scope('embedding'):
                # this is actually the matrix (v_c) for c=1,...,V in the document
                self._embed_matrix = tf.Variable(tf.random_uniform(shape = [self._vocabs_size, self._embed_dim],
                                                                   minval=-1.0, maxval=1.0), name = 'embed_matrix')
                self._embed = tf.nn.embedding_lookup(self._embed_matrix,
                                                     self._center_words, name = 'embed_center_word')

    def _create_neg_sampling_loss(self):
        with self._graph.as_default():
            self._softmax_weights = tf.Variable(tf.truncated_normal(shape = [self._vocabs_size, self._embed_dim],
                                                                    stddev=1.0 / np.sqrt(self._embed_dim)), name = 'softmax_w')
            self._softmax_biases = tf.Variable(tf.zeros([self._vocabs_size]), name = 'softmax_b')

            labels_matrix = tf.cast(self._target_words, dtype=tf.int64)

            # negative sampling.
            sampled_values = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=self._nb_neg_sample,
                unique=True,
                range_max=self._vocabs_size,
                distortion=0.75,
                unigrams=self._freqs_list))

            # compute loss
            self._loss = tf.nn.sampled_softmax_loss(self._softmax_weights, self._softmax_biases,
                                                    self._target_words, self._embed,
                                                    self._nb_neg_sample, self._vocabs_size,
                                                    sampled_values=sampled_values)

            # compute cost
            self._cost = tf.reduce_mean(self._loss)

            # creat optimizer
            self._optimizer = tf.train.AdamOptimizer().minimize(self._cost)


    def build_graph(self):
        self._graph = tf.Graph()

        # create input/output placeholder
        self._create_placeholder()

        # create embedding layers
        self._create_embedding()

        # create neg-sampling loss & optimizer
        self._create_neg_sampling_loss()

    def get_batches(self):
        n_batches = len(self._train_wordids) // self._batch_size

        # only full batches
        words = self._train_wordids[:n_batches * self._batch_size]

        for idx in range(0, len(words), self._batch_size):
            x, y = [], []
            batch = words[idx:idx + self._batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = get_target(batch, ii, self._window_size)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield x, y

    def train(self):
        epochs = 5
        iteration = 1
        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())

            loss = 0
            for e in range(1, epochs + 1):
                batches = self.get_batches()
                start = time()
                for x, y in batches:
                    feed = {self._center_words : x,
                            self._target_words : np.array(y)[:, None]}
                    train_loss, _ = sess.run([self._cost,
                                              self._optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time()
                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/batch".format((end - start) / 100))
                        loss = 0
                        start = time()

                    iteration+=1