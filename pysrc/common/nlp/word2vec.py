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


class Word2vecSampling(object):
    def __init__(self,
                 vocabs,
                 word2id,
                 id2word,
                 freqs,
                 train_wordids):
        self._vocabs = vocabs
        self._word2id = word2id
        self._id2word = id2word
        self._freqs = freqs

        self._train_wordids = train_wordids

        self._vocabs_size = len(self._vocabs)
        self._freqs_list = [0.] * self._vocabs_size
        for w in self._vocabs:
            self._freqs_list[self._word2id[w]] = self._freqs[w]

    def _create_placeholder(self):
        with self._graph.as_default():
            # first dim is batch-size can be variable so we set it to None
            self._center_words = tf.placeholder(tf.int64, shape=[None],    name='center_words')
            self._target_words = tf.placeholder(tf.int64, shape=[None, 1], name='target_words')

    def _create_embedding(self):
        with self._graph.as_default():
            with tf.name_scope('embedding'):
                # this is actually the matrix (v_c) for c=1,...,V in the document
                self._embed_matrix = tf.Variable(tf.random_uniform(shape = [self._vocabs_size, self._embed_dim],
                                                                   minval=-1.0, maxval=1.0), name = 'embed_matrix')
                self._embed = tf.nn.embedding_lookup(self._embed_matrix,
                                                     self._center_words, name = 'embed_center_word')

    def _sampled_logit(self, sampled_ids, true_expected_count, sampled_expected_count):
        with self._graph.as_default():

            # we need flatten target so that when we do look-up we obtain output of shape (batch_size x embed_dim)
            _target_flat = tf.reshape(self._target_words, [-1])

            # Weights for labels: [None, emb_dim]
            true_w = tf.nn.embedding_lookup(self._softmax_weights, _target_flat)

            # Biases for labels: [None]
            true_b = tf.nn.embedding_lookup(self._softmax_biases, _target_flat)

            # self._embed: [None, emb_dim], true_w: [None, emb_dim]
            # true logits: [None, 1]
            true_logits = tf.reshape(tf.reduce_sum(tf.multiply(self._embed, true_w), 1) + true_b, [-1, 1])

            assert (true_logits.get_shape().as_list() == [None, 1])

            # Weights for sampled ids: [num_sampled, emb_dim]
            sampled_w = tf.nn.embedding_lookup(self._softmax_weights, sampled_ids)

            # Biases for sampled ids: [num_sampled]
            sampled_b = tf.nn.embedding_lookup(self._softmax_biases, sampled_ids)

            # sample logits [None, num_sampled]
            sampled_logits = tf.matmul(self._embed, sampled_w, transpose_b=True) + sampled_b

            assert (true_logits.get_shape().as_list() == [None, self._nb_neg_sample])

            if self._subtract_log_q:
                # Subtract log of Q(l), prior probability that l appears in sampled.
                true_logits -= tf.log(true_expected_count)
                sampled_logits -= tf.log(sampled_expected_count)

            out_logits = tf.concat([true_logits, sampled_logits], 1)
            out_labels = tf.concat([tf.ones_like(true_logits), tf.zeros_like(sampled_logits)], 1)

        return out_logits, out_labels

    def _create_neg_sampling_loss(self):
        with self._graph.as_default():
            self._softmax_weights = tf.Variable(tf.truncated_normal(shape = [self._vocabs_size, self._embed_dim],
                                                                    stddev=1.0 / np.sqrt(self._embed_dim)), name = 'softmax_w')
            self._softmax_biases = tf.Variable(tf.zeros([self._vocabs_size]), name = 'softmax_b')

            # negative sampling.
            sampled_ids = None
            true_expected_count = None
            sampled_expected_count = None

            if self._sampling_method == 'neg':
                sampled_ids, true_expected_count, sampled_expected_count = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=self._target_words,
                    num_true=1,
                    num_sampled=self._nb_neg_sample,
                    unique=True,
                    range_max=self._vocabs_size,
                    distortion=0.75,
                    unigrams=self._freqs_list)
            elif self._sampling_method == 'log_uni':
                sampled_ids, true_expected_count, sampled_expected_count = tf.nn.log_uniform_candidate_sampler(
                    true_classes=self._target_words,
                    num_true=1,
                    num_sampled=self._nb_neg_sample,
                    unique=True,
                    range_max=self._vocabs_size)

            out_logits, out_labels = self._sampled_logit(sampled_ids, true_expected_count, sampled_expected_count)

            # compute loss
            if self._loss_func == 'sampled_softmax':
                self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=out_labels, logits=out_logits)
            elif self._loss_func == 'nce':
                self._loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=out_labels, logits=out_logits)
            else:
                raise Exception('Unknown sampling method {}, currently only support neg/nce'.format(self._sampling_method))

            # compute cost
            self._cost = tf.reduce_mean(self._loss)

            # create global_step to store training-step
            self._global_step = tf.Variable(0, dtype=tf.int32,
                                            trainable=False, name='global_step')

            # creat optimizer and pass the global_step to make it incremented each traingin-step
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._cost,
                                                                                                 global_step=self._global_step)


    def build_graph(self,settings = {'embed_dim'       : 200,
                                     'nb_neg_sample'   : 100,
                                     'learning_rate'   : 0.2,
                                     'sampling_method' : 'neg',
                                     'loss_func'       : 'nce',
                                     'subtract_log_q' :  True}):
        # get hyper-parameters
        self._embed_dim         = settings.get('embed_dim', 200)
        self._nb_neg_sample     = settings.get('nb_neg_sample', 100)
        self._learning_rate     = settings.get('learning_rate', 0.2),
        self._sampling_method   = settings.get('sampling_method', 'neg')
        self._loss_func         = settings.get('loss_func', 'nce')
        self._subtract_log_q    = settings.get('subtract_log_q', True)
        assert (self._sampling_method == 'neg' or self._sampling_method == 'log_uni')
        assert (self._loss_func == 'sampled_softmax' or self._loss_func == 'nce')

        self._graph = tf.Graph()

        # create input/output placeholder
        self._create_placeholder()

        # create embedding layers
        self._create_embedding()

        # create neg-sampling loss & optimizer
        self._create_neg_sampling_loss()

        # create saver to save
        with self._graph.as_default():
            self._saver = tf.train.Saver()

    def get_batches(self, batch_size, window_size):
        n_batches = len(self._train_wordids) // batch_size

        # only full batches
        words = self._train_wordids[:n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx + batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield x, y

    def train(self, epochs, batch_size, window_size,
                    print_every = 100, save_every=1000):
        iteration = 1
        n_batches = len(self._train_wordids) // batch_size

        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())

            loss = 0
            for e in range(1, epochs + 1):
                batches = self.get_batches(batch_size, window_size)
                start = time()
                i_batch = 1
                for x, y in batches:
                    feed = {self._center_words : x,
                            self._target_words : np.array(y)[:, None]}
                    train_loss, _ = sess.run([self._cost,
                                              self._optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % print_every == 0:
                        end = time()
                        print("Epoch ({}/{})".format(e, epochs),
                              "Batch ({:>5d}/{:<5d})".format(i_batch, n_batches),
                              "Iteration: {:>8d}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/batch".format((end - start) / 100))
                        loss = 0
                        start = time()

                    if iteration % save_every == 0:
                        self._saver.save(sess,
                                         'checkpoints/skip-gram',
                                         global_step=self._global_step)

                    i_batch += 1
                    iteration+=1
