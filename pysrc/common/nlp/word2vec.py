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

def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = tf.shape(x)[1]
  ones_shape = tf.stack([cols, 1])
  ones = tf.ones(ones_shape, x.dtype)
  return tf.reshape(tf.matmul(x, ones), [-1])


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

    def _sampled_logit(self, sampled_values):
        with self._graph.as_default():
            sampled_ids, true_expected_count, sampled_expected_count = sampled_values

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

            assert (sampled_logits.get_shape().as_list() == [None, self._nb_neg_sample])

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
            sampled_values = None

            if self._sampling_method == 'fixed_unigram':
                sampled_values = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=self._target_words,
                    num_true=1,
                    num_sampled=self._nb_neg_sample,
                    unique=True,
                    range_max=self._vocabs_size,
                    distortion=0.75,
                    unigrams=self._freqs_list)
            elif self._sampling_method == 'log_uniform':
                sampled_values = tf.nn.log_uniform_candidate_sampler(
                    true_classes=self._target_words,
                    num_true=1,
                    num_sampled=self._nb_neg_sample,
                    unique=True,
                    range_max=self._vocabs_size)

            if self._use_tf_loss:
                # compute loss
                if self._loss_func == 'sampled_softmax':
                    self._loss = tf.nn.sampled_softmax_loss(self._softmax_weights, self._softmax_biases,
                                                            self._target_words, self._embed,
                                                            self._nb_neg_sample, self._vocabs_size, sampled_values=sampled_values)
                else:
                    self._loss = tf.nn.nce_loss(self._softmax_weights, self._softmax_biases,
                                                self._target_words, self._embed,
                                                self._nb_neg_sample, self._vocabs_size,
                                                sampled_values=sampled_values)

            else:
                out_logits, out_labels = self._sampled_logit(sampled_values)

                # compute loss
                if self._loss_func == 'sampled_softmax':
                    self._loss = tf.nn.softmax_cross_entropy_with_logits(labels=out_labels, logits=out_logits)
                elif self._loss_func == 'nce':
                    self._loss = _sum_rows(tf.nn.sigmoid_cross_entropy_with_logits(labels=out_labels,
                                                                                   logits=out_logits))
                else:
                    raise Exception('Unknown sampling method {}, currently only support neg/nce'.format(self._sampling_method))


            # compute cost
            self._cost = tf.reduce_mean(self._loss)

            tf.summary.scalar("training_loss", self._cost)

            # create global_step to store training-step
            self._global_step = tf.Variable(0, dtype=tf.int32,
                                            trainable=False, name='global_step')

            # creat optimizer and pass the global_step to make it incremented each traingin-step
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._cost,
                                                                                                 global_step=self._global_step)


    def build_graph(self,settings = {'embed_dim'       : 200,
                                     'nb_neg_sample'   : 100,
                                     'learning_rate'   : 0.2,
                                     'sampling_method' : 'fixed_unigram',
                                     'loss_func'       : 'nce',
                                     'use_tf_loss'     : False,
                                     'subtract_log_q'  : True
                                     }):
        # get hyper-parameters
        self._embed_dim         = settings.get('embed_dim',         200)
        self._nb_neg_sample     = settings.get('nb_neg_sample',     100)
        self._learning_rate     = settings.get('learning_rate',     0.2)
        self._sampling_method   = settings.get('sampling_method',   'fixed_unigram')
        self._loss_func         = settings.get('loss_func',         'nce')
        self._subtract_log_q    = settings.get('subtract_log_q',    True)
        self._use_tf_loss       = settings.get('use_tf_loss',       False)

        assert (self._sampling_method == 'fixed_unigram' or self._sampling_method == 'log_uniform')
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

    def train(self, epochs, batch_size, window_size, max_iters = None,
                    print_every = 100, save_every=1000,
                    summary_every = 5, summary_path = None):
        iteration = 1
        n_batches = len(self._train_wordids) // batch_size

        with tf.Session(graph=self._graph) as sess:
            sess.run(tf.global_variables_initializer())

            summary_op = None
            summary_writer = None

            run_ops = [self._cost, self._optimizer]
            if summary_path != None:
                summary_op = tf.summary.merge_all()
                save_path = 'logs/{}/run(lr={},lf={},sampling={},use_tf={})'.format( summary_path,
                                                                                    self._learning_rate,
                                                                                    self._loss_func,
                                                                                    self._sampling_method,
                                                                                    self._use_tf_loss)
                summary_writer = tf.summary.FileWriter(save_path, self._graph)
                run_ops.append(summary_op)

            loss = 0
            last_summary_time = 0
            break_loop = False
            t0 = time()
            for e in range(1, epochs + 1):
                batches = self.get_batches(batch_size, window_size)
                start = time()
                i_batch = 1
                for x, y in batches:
                    feed = {self._center_words : x,
                            self._target_words : np.array(y)[:, None]}

                    output = sess.run(run_ops, feed_dict=feed)

                    loss += output[0]

                    if iteration % print_every == 0:
                        end = time()
                        print("Epoch ({}/{})".format(e, epochs),
                              "Batch ({:>5d}/{:<5d})".format(i_batch, n_batches),
                              "Iteration: {:>8d}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/batch".format((end - start) / 100))
                        loss = 0

                        start = time()
                    if (summary_writer != None) and (time() - last_summary_time > summary_every):
                        summary_writer.add_summary(output[-1],
                                                   self._global_step.eval())

                    if iteration % save_every == 0:
                        checkpoint_string = 'checkpoints/sg(lr={},lf={},sampling={},use_tf={})'.format(self._learning_rate,
                                                                                                       self._loss_func,
                                                                                                       self._sampling_method,
                                                                                                       self._use_tf_loss)
                        self._saver.save(sess,
                                         checkpoint_string,
                                         global_step=self._global_step)

                    i_batch   += 1
                    iteration += 1

                    if (max_iters!= None and iteration > max_iters):
                        break_loop = True
                        break

                if break_loop:
                    break

            # save last checkpoint
            if not break_loop:
                checkpoint_string = 'checkpoints/sg_lr={},lf={},sampling={},use_tf={}'.format(self._learning_rate,
                                                                                              self._loss_func,
                                                                                              self._sampling_method,
                                                                                              self._use_tf_loss)
                self._saver.save(sess,
                                 checkpoint_string,
                                 global_step=self._global_step)

            print('Total run-time {:.2f}'.format(time() - t0))

    def build_eval_graph(self):
        with self._graph.as_default():
            analogy_a = tf.placeholder(tf.int32)
            analogy_b = tf.placeholder(tf.int32)
            analogy_c = tf.placeholder(tf.int32)

            # normalized word embeddings of shape [vocab_size, emb_dim]
            nemb = tf.nn.l2_normalize(self._embed_matrix, 1)

            # each row of word embeddings
            a_emb = tf.gather(nemb, analogy_a)
            b_emb = tf.gather(nemb, analogy_b)
            c_emb = tf.gather(nemb, analogy_c)

            # we expect target = c_emb + (b_emb - a_emb): has shape [N, emb_dim]
            target = c_emb + (b_emb - a_emb)

            # compute the cosine distance: has shape [N, vocab_size]
            dist = tf.matmul(target, nemb, transpose_b=True)

            # for each equestion pick top 4 words
            _, pred_idx = tf.nn.top_k(dist, 4)

            # nodes for computing 10-neighbors for a given word
            nearby_word = tf.placeholder(tf.int32)
            nearby_emb = tf.gather(nemb, nearby_word)
            nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
            nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, 10)

            # put to self so that we can use it other function
            self._analogy_a = analogy_a
            self._analogy_b = analogy_b
            self._analogy_c = analogy_c
            self._analogy_pred_idx = pred_idx
            self._nearby_word = nearby_word
            self._nearby_val = nearby_val
            self._nearby_idx = nearby_idx

    def _predict(self, analogy, sess):
        """Predict the top 4 answers for analogy questions."""
        idx, = sess.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def load_checkpoint(self, checkpoint):
        sess = tf.Session(graph=self._graph)
        self._saver.restore(sess, checkpoint)
        return sess

    def analogy(self, sess, w0, w1, w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        print('predict {}-{} as {}-?'.format(w0, w1, w2))
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid, sess)
        nb_analogy = 0
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print('answer: {}\n'.format(c))
                nb_analogy += 1
                break

        if (nb_analogy == 0):
            print("unknown")

    def nearby(self, sess, w):
        """Prints out nearby words given a list of words."""
        ids = np.array([self._word2id.get(w, 0) ])
        vals, idx = sess.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})

        print("\nNearest neighbours of [{}]\n=====================================".format(w))
        for (neighbor, distance) in zip(idx[0, :], vals[0, :]):
            print("%-20s %6.4f" % (self._id2word[neighbor], distance))

